from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
import json
import re
import string
import sys
import platform
import subprocess
import sqlite3
from pathlib import Path
from typing import Any

from Agent.evaluation.checkpoints import append_checkpoint, load_checkpoints
from Agent.evaluation.config import resolve_path
from Agent.evaluation.direct_dataset import EvaluationRecord, load_evaluation_dataset
from Agent.evaluation.io import write_json, write_jsonl
from Agent.evaluation.models import AgentOutput
from Agent.evaluation.capability_metrics import event_metrics, reasoning_efficiency, state_consistency
from Agent.evaluation.io import file_hash
from Agent.evaluation.reasoning_steps import aggregate_step_accuracy
from Agent.query_engine import QueryEngine, QueryOptions


def _runtime_for(config: dict[str, Any], project_root: Path, profile_name: str) -> Any:
    kind = config["runtime"]["kind"]
    if kind == "llm_direct":
        from Agent.runtime import DirectRuntime
        return DirectRuntime(config, project_root, profile_name)
    if kind == "bfcl_protocol":
        from Agent.runtime import BFCLProtocolRuntime
        return BFCLProtocolRuntime(config, project_root, profile_name)
    if kind == "qwen_harness":
        from Agent.runtime import QwenHarnessRuntime
        return QwenHarnessRuntime(config, project_root, profile_name)
    raise ValueError(f"unknown runtime kind: {kind}")


async def execute_direct_run(
    config: dict[str, Any], config_path: Path, project_root: Path, *, limit: int | None = None,
    resume: bool = False, rescore: bool = False, profile: str | None = None, runtime: Any | None = None,
    run_name: str | None = None,
) -> Path:
    dataset_path = resolve_path(config["dataset"]["path"], config_path, project_root)
    records = load_evaluation_dataset(dataset_path, project_root / "Agent", config)
    records = records[:limit] if limit is not None else records
    profile_name = profile or config["model"]["default_profile"]
    if profile_name not in config["model"]["profiles"]:
        raise ValueError(f"unknown model profile: {profile_name}")
    selected_profile = config["model"]["profiles"][profile_name]
    run_dir = project_root / "runs" / (run_name or f"{config_path.stem}-{profile_name}")
    checkpoint = run_dir / "checkpoints/agent_outputs.jsonl"
    if run_dir.exists() and not (resume or rescore) and any(path.is_file() for path in run_dir.rglob("*")):
        raise FileExistsError(f"run directory already contains results: {run_dir}; use --resume or --rescore")
    if not run_dir.exists() and (resume or rescore):
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    selected = {record.input["id"] for record in records}

    if rescore:
        output_path = run_dir / "agent_outputs.jsonl"
        outputs = {output.id: output for output in _read_outputs(output_path if output_path.exists() else checkpoint)}
    else:
        outputs = load_checkpoints(checkpoint) if resume else {}
        # Validate model dependencies and endpoint configuration before creating
        # any run artifacts, so a startup failure does not poison the run name.
        active_runtime = runtime or _runtime_for(config, project_root, profile_name)
        if hasattr(active_runtime, "run_state_root"):
            active_runtime.run_state_root = run_dir
        if hasattr(active_runtime, "bfcl_executor"):
            active_runtime.bfcl_executor.result_root = run_dir / "bfcl_memory_state"
        run_dir.mkdir(parents=True, exist_ok=True)
        query_engine = QueryEngine(active_runtime, state_root=run_dir / "query_engine")

        is_bfcl_memory = any(
            record.input["metadata"].get("category", "").startswith("memory_") for record in records
        )
        semaphore = asyncio.Semaphore(1 if is_bfcl_memory else 4)

        async def process(record: EvaluationRecord) -> AgentOutput:
            async with semaphore:
                output = await query_engine.submit(record.input, QueryOptions(
                    session_id=record.input["id"], message_id=record.input["id"], persist_session=False,
                ))
                append_checkpoint(checkpoint, output)
                return output

        pending = [record for record in records if record.input["id"] not in outputs]
        if is_bfcl_memory:
            completed = [await process(record) for record in pending]
        else:
            completed = await asyncio.gather(*(process(record) for record in pending))
        outputs.update({output.id: output for output in completed})
    missing = selected - outputs.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} outputs")
    ordered = [outputs[record.input["id"]] for record in records]
    results = [_score(config, record, outputs[record.input["id"]], project_root) for record in records]
    metrics = _aggregate(config, results)
    write_json(run_dir / "resolved_config.json", {"config": config, "profile": profile_name, "dataset_path": str(dataset_path)})
    write_jsonl(run_dir / "agent_outputs.jsonl", (output.to_dict() for output in ordered))
    write_jsonl(run_dir / "sample_results.jsonl", results)
    write_json(run_dir / "metrics.json", metrics)
    output_values = [output.to_dict() for output in ordered]
    write_jsonl(run_dir / "samples.jsonl", ({"id": record.input["id"], "input": record.input} for record in records))
    write_jsonl(run_dir / "trajectories.jsonl", (
            {"sample_id": output.id, "trajectory_id": output.metadata.get("trajectory_id"),
             "runtime_kind": output.metadata.get("runtime"),
             "provider": selected_profile.get("provider", "openai_compatible"),
             "model_profile": profile_name, "exact_model_name": selected_profile["name"],
             "feature_set": config["features"], "status": output.status,
             "usage": output.usage, "timing": output.timing}
            for output in ordered
    ))
    write_jsonl(run_dir / "metric_details.jsonl", results)
    metrics["reasoning_efficiency"] = reasoning_efficiency(results, output_values)
    metrics["step_accuracy"] = aggregate_step_accuracy(output_values)
    event_path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()] if event_path.exists() else []
    outcomes = {output.metadata.get("trajectory_id"): bool(result.get("correct"))
                for output, result in zip(ordered, results) if output.metadata.get("trajectory_id")}
    metrics["capabilities"] = event_metrics(
        events, outcomes, planning_enabled=config["features"].get("planning", {}).get("enabled", False)
    )
    if config["dataset"]["name"] == "bfcl":
        metrics["capabilities"]["tools"].update(_bfcl_component_metrics(records, ordered))
    metrics["verification"] = _verification_metrics(config, records, ordered, results, project_root)
    if config["dataset"]["name"] == "locomo":
        metrics["memory"] = _memory_metrics(records, ordered, results)
        mode = next((output.metadata.get("locomo_mode") for output in ordered
                     if output.metadata.get("locomo_mode")), "persistent_memory")
        metrics["context_or_memory_mode"] = mode
    metrics["state"] = state_consistency(run_dir / "checkpoints.jsonl")
    metrics["state"]["state_recovery_rate"] = None
    metrics["state"]["state_recovery_unavailable_reason"] = "no fault-injection recovery trials in this run"
    metrics["unavailable"] = {
        "step_accuracy_semantic": "semantic step judgments are unavailable; deterministic arithmetic coverage is reported separately",
        "planning_efficiency": "verified shortest-path annotations are not present",
        "search_recall_mrr": "complete relevance judgments are not present",
        "memory_update_accuracy": "required-update annotations are not present",
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", _run_manifest(config, config_path, dataset_path, profile_name, project_root,
                                                           experiment_id=run_dir.name))
    if config["dataset"]["name"] == "bfcl":
        write_jsonl(run_dir / "bfcl_results.jsonl", (_bfcl_submission(record, outputs[record.input["id"]]) for record in records))
    return run_dir


def _run_manifest(config: dict[str, Any], config_path: Path, dataset_path: Path,
                  profile_name: str, project_root: Path, *, experiment_id: str) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    profile = config["model"]["profiles"][profile_name]
    ollama = None
    if profile.get("provider") == "ollama":
        try:
            version = subprocess.run(["ollama", "--version"], capture_output=True, text=True,
                                     timeout=5, check=True).stdout.strip()
            shown = subprocess.run(["ollama", "show", "--modelfile", profile["name"]], capture_output=True,
                                   text=True, timeout=10, check=True).stdout
            ollama = {"version": version, "model_definition_sha256": __import__("hashlib").sha256(shown.encode()).hexdigest()}
        except Exception as exc:
            ollama = {"available": False, "error": type(exc).__name__}
    return {
        "schema_version": "1.0", "experiment_id": experiment_id,
        "runtime_kind": config["runtime"]["kind"],
        "provider": profile.get("provider", "openai_compatible"),
        "model_profile": profile_name, "exact_model_name": profile["name"],
        "model": profile, "ollama": ollama, "features": config["features"],
        "feature_set": config["features"],
        "resource_budget": config["limits"], "resolved_config": config,
        "config_sha256": file_hash(config_path), "dataset_sha256": file_hash(dataset_path),
        "code_commit": commit, "normalized_tokenizer": "unicode-wordpunct-v1",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(), "concurrency": 4},
    }


def _verification_metrics(config: dict[str, Any], records: list[EvaluationRecord], outputs: list[AgentOutput],
                          results: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    final_by_id = {row["id"]: bool(row.get("correct")) for row in results}
    records_by_id = {record.input["id"]: record for record in records}
    evaluated = 0
    tp = fp = fn = corrected = regressions = initially_correct = 0
    transitions = {"incorrect_to_correct": 0, "incorrect_to_incorrect": 0,
                   "correct_to_correct": 0, "correct_to_incorrect": 0}
    for output in outputs:
        verification = output.metadata.get("verification_result")
        if not verification:
            continue
        before_output = AgentOutput(output.id, "success", verification.get("answer_before"))
        before_correct = bool(_score(config, records_by_id[output.id], before_output, project_root).get("correct"))
        after_correct = final_by_id[output.id]
        detected = bool(verification.get("error_detected"))
        evaluated += 1
        tp += int(detected and not before_correct)
        fp += int(detected and before_correct)
        fn += int(not detected and not before_correct)
        corrected += int(detected and not before_correct and after_correct)
        initially_correct += int(before_correct)
        regressions += int(before_correct and not after_correct)
        transitions[("correct" if before_correct else "incorrect") + "_to_" +
                    ("correct" if after_correct else "incorrect")] += 1
    return {
        "evaluated": evaluated,
        "error_detection_recall": tp / (tp + fn) if tp + fn else None,
        "error_detection_precision": tp / (tp + fp) if tp + fp else None,
        "error_correction_rate": corrected / tp if tp else None,
        "correction_regression_rate": regressions / initially_correct if initially_correct else None,
        "transitions": transitions,
    }


def _memory_metrics(records: list[EvaluationRecord], outputs: list[AgentOutput],
                    results: list[dict[str, Any]]) -> dict[str, Any]:
    record_by_id = {record.input["id"]: record for record in records}
    result_by_id = {row["id"]: row for row in results}
    relevant_total = retrieved_relevant = 0
    stored_bytes = stored_tokens = correct_answers = 0
    mapped_questions = total_questions = 0
    for output in outputs:
        result = result_by_id.get(output.id, {})
        correct_answers += sum(bool(row.get("correct")) for row in result.get("question_results", []))
        total_questions += len(result.get("question_results", []))
        path_value = output.metadata.get("memory_db")
        if path_value and Path(path_value).exists():
            path = Path(path_value)
            stored_bytes += path.stat().st_size
            connection = sqlite3.connect(path)
            try:
                contents = [row[0] for row in connection.execute("SELECT content FROM memories")]
                stored_tokens += sum(len(re.findall(r"\w+|[^\w\s]", content, flags=re.UNICODE)) for content in contents)
            finally:
                connection.close()
        reference_questions = {row["id"]: row for row in (record_by_id[output.id].reference or {}).get("questions", [])}
        tool_results = {message.get("tool_call_id"): message.get("content", "") for message in output.messages
                        if message.get("role") == "tool"}
        for call in output.tool_calls:
            if call.get("phase") != "question" or call.get("name") != "memory_search":
                continue
            reference = reference_questions.get(call.get("question_id"), {})
            evidence = {str(value) for value in reference.get("evidence", [])}
            if not evidence:
                continue
            mapped_questions += 1
            relevant_total += len(evidence)
            try:
                rows = json.loads(tool_results.get(call.get("id"), "[]"))
                retrieved = {str(source) for row in rows for source in json.loads(row.get("source_ids", "[]"))}
            except (ValueError, TypeError):
                retrieved = set()
            retrieved_relevant += len(evidence & retrieved)
    return {
        "memory_qa_correct": correct_answers, "memory_qa_total": total_questions,
        "memory_qa_accuracy": correct_answers / total_questions if total_questions else None,
        "retrieval_recall": retrieved_relevant / relevant_total if relevant_total else None,
        "mapped_question_coverage": mapped_questions / total_questions if total_questions else 0.0,
        "stored_tokens": stored_tokens, "stored_bytes": stored_bytes,
        "storage_efficiency_per_token": correct_answers / stored_tokens if stored_tokens else None,
        "storage_efficiency_per_byte": correct_answers / stored_bytes if stored_bytes else None,
        "memory_update_accuracy": None,
        "memory_update_unavailable_reason": "required-update annotations are not present",
    }


def _bfcl_component_metrics(records: list[EvaluationRecord], outputs: list[AgentOutput]) -> dict[str, Any]:
    output_by_id = {output.id: output for output in outputs}
    selection_total = selection_correct = argument_total = argument_correct = format_correct = 0
    skipped_multiturn = 0
    for record in records:
        output = output_by_id[record.input["id"]]
        calls = output.tool_calls
        format_correct += int(all(call.get("name") and isinstance(call.get("args"), dict) for call in calls))
        if record.input["task_family"] != "single_turn" or record.input["metadata"].get("category", "").startswith("web_search"):
            skipped_multiturn += 1
            continue
        category = record.input["metadata"].get("category", "")
        if "irrelevance" in category:
            selection_total += 1
            selection_correct += int(not calls)
            continue
        ground_truth = (record.reference or {}).get("ground_truth") or []
        if not ground_truth:
            skipped_multiturn += 1
            continue
        selection_total += 1
        expected = [next(iter(candidate.items())) for candidate in ground_truth if len(candidate) == 1]
        actual_names = [call.get("name") for call in calls]
        expected_names = [name for name, _ in expected]
        unordered = "parallel" in category
        selection_ok = (sorted(actual_names) == sorted(expected_names) if unordered else actual_names == expected_names)
        selection_correct += int(selection_ok)
        remaining = list(expected)
        for call in calls:
            match_index = next((index for index, (name, _) in enumerate(remaining) if name == call.get("name")), None)
            if match_index is None:
                continue
            _, possibilities = remaining.pop(match_index)
            argument_total += 1
            args = call.get("args", {})
            valid = not (set(args) - set(possibilities))
            for key, accepted in possibilities.items():
                if key not in args:
                    valid = valid and "" in accepted
                else:
                    valid = valid and args[key] in accepted
            argument_correct += int(valid)
    return {
        "format_valid_samples": format_correct, "format_evaluated_samples": len(records),
        "format_accuracy": format_correct / len(records) if records else None,
        "tool_selection_correct": selection_correct, "tool_selection_evaluated": selection_total,
        "tool_selection_accuracy": selection_correct / selection_total if selection_total else None,
        "argument_correct": argument_correct, "argument_evaluated": argument_total,
        "argument_accuracy": argument_correct / argument_total if argument_total else None,
        "call_level_unavailable_samples": skipped_multiturn,
    }


def _score(config: dict[str, Any], record: EvaluationRecord, output: AgentOutput, project_root: Path) -> dict[str, Any]:
    category = record.input["metadata"].get("category", "main")
    if config["dataset"]["name"] == "locomo":
        return _score_locomo(record, output)
    if output.status != "success":
        return {"id": output.id, "status": output.status, "score": 0.0, "correct": False,
                "metadata": {"category": category}, "error": output.error}
    if config["dataset"]["name"] == "gsm8k":
        predicted = _last_number(output.final_answer or "")
        expected = _last_number((record.reference or {}).get("answer", ""))
        correct = _numeric_equivalent(predicted, expected)
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct), "correct": correct,
                "prediction": predicted, "reference": expected, "comparison": "decimal_exact"}
    if config["dataset"]["name"] == "aime":
        predicted = _aime_integer(output.final_answer or "")
        expected = int((record.reference or {})["answer"])
        correct = predicted == expected
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct),
                "correct": correct, "prediction": predicted, "reference": expected,
                "metadata": {"category": category}}
    if config["dataset"]["name"] == "competition_math":
        predicted = _math_answer(output.final_answer or "")
        expected = _math_answer((record.reference or {}).get("solution", ""))
        correct = predicted is not None and expected is not None and _math_equivalent(predicted, expected)
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct),
                "correct": correct, "prediction": predicted, "reference": expected,
                "metadata": {"category": category, "level": record.input["metadata"].get("level")}}
    if config["dataset"]["name"] == "gaia":
        predicted = _gaia_normalize(output.final_answer or "")
        expected = _gaia_normalize((record.reference or {}).get("answer", ""))
        correct = predicted == expected
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct),
                "correct": correct, "prediction": predicted, "reference": expected,
                "metadata": {"category": category, "level": record.input["metadata"].get("level")}}
    return _score_bfcl(record, output, project_root)


def _score_locomo(record: EvaluationRecord, output: AgentOutput) -> dict[str, Any]:
    predictions = {row["id"]: row.get("answer", "") for row in output.metadata.get("locomo_answers", [])}
    question_results: list[dict[str, Any]] = []
    for reference in (record.reference or {}).get("questions", []):
        prediction = predictions.get(reference["id"], "")
        category = int(reference["category"])
        if output.status != "success":
            score = 0.0
        elif category == 5:
            lowered = prediction.strip().lower()
            score = float("no information available" in lowered or "not mentioned" in lowered)
        else:
            answer = str(reference.get("answer", ""))
            if category == 3:
                answer = answer.split(";", 1)[0].strip()
            score = _locomo_multi_answer_f1(prediction, answer) if category == 1 else _locomo_f1(prediction, answer)
        question_results.append({
            "id": reference["id"], "category": category, "score": score,
            "correct": score == 1.0, "prediction": prediction,
            "reference": reference.get("answer"), "evidence": reference.get("evidence", []),
        })
    mean_score = sum(row["score"] for row in question_results) / len(question_results)
    return {"id": output.id, "status": output.status, "score": mean_score,
            "correct": all(row["correct"] for row in question_results),
            "metadata": {"category": "all", "conversation_id": record.input["metadata"]["conversation_id"]},
            "question_results": question_results, "error": output.error}


def _score_bfcl(record: EvaluationRecord, output: AgentOutput, project_root: Path) -> dict[str, Any]:
    package = project_root / "benchmark/BFCL-main/berkeley-function-call-leaderboard"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    category = record.input["metadata"]["category"]
    reference = record.reference or {}
    try:
        if reference.get("evaluator") == "non_scoring_prerequisite":
            return {"id": output.id, "status": output.status, "score": None, "correct": None,
                    "scored": False, "metadata": {"category": category}}
        if category.startswith("web_search") or (category.startswith("memory_") and not category.endswith("_prereq")):
            from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
            result = agentic_checker(output.final_answer or "", reference["ground_truth"])
            correct = bool(result.get("valid"))
        elif record.input["task_family"] == "multiturn":
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
            prompt = {"id": record.input["id"], **record.input["environment"]}
            result = multi_turn_checker(output.metadata.get("bfcl_calls_by_turn", []), reference["ground_truth"], prompt, category, "direct")
            correct = bool(result.get("valid"))
        elif "irrelevance" in category:
            correct = not output.tool_calls
            result = {"valid": correct}
        elif not reference:
            correct = bool(output.tool_calls)
            result = {"valid": correct, "note": "relevance category has no AST ground truth"}
        else:
            decoded = [{call["name"]: call["args"]} for call in output.tool_calls]
            try:
                from bfcl_eval.constants.enums import Language
                from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
                language = Language.JAVA if category == "simple_java" else Language.JAVASCRIPT if category == "simple_javascript" else Language.PYTHON
                result = ast_checker(record.input["tools"], decoded, reference["ground_truth"], language, category, "gpt-4o-2024-11-20-FC")
                correct = bool(result.get("valid"))
            except ModuleNotFoundError:
                correct = _fallback_ast_match(decoded, reference["ground_truth"], unordered="parallel" in category)
                result = {"valid": correct, "checker": "dependency-free BFCL-compatible fallback"}
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct), "correct": correct,
                "metadata": {"category": category}, "details": result}
    except Exception as exc:
        return {"id": output.id, "status": "evaluator_error", "score": 0.0, "correct": False,
                "metadata": {"category": category}, "error": {"type": type(exc).__name__, "message": str(exc)}}


def _aggregate(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    if config["dataset"]["name"] == "locomo":
        questions = [question for result in results for question in result["question_results"]]
        by_category: dict[str, dict[str, int | float]] = {}
        for question in questions:
            key = str(question["category"])
            group = by_category.setdefault(key, {"total": 0, "score_sum": 0.0, "mean_f1": 0.0})
            group["total"] += 1
            group["score_sum"] += question["score"]
        for group in by_category.values():
            group["mean_f1"] = group["score_sum"] / group["total"]
        total_score = sum(question["score"] for question in questions)
        return {"schema_version": "3.0", "benchmark": "locomo", "total": len(questions),
                "score_sum": total_score, "mean_f1": total_score / len(questions),
                "by_category": by_category, "conversation_count": len(results),
                "sampling_rate": 0.10}
    scored_results = [result for result in results if result.get("scored", True)]
    correct = sum(bool(result["correct"]) for result in scored_results)
    by_category: dict[str, dict[str, int | float]] = {}
    for result in scored_results:
        category = result.get("metadata", {}).get("category", "main")
        group = by_category.setdefault(category, {"total": 0, "correct": 0, "accuracy": 0.0})
        group["total"] += 1
        group["correct"] += int(result["correct"])
    for group in by_category.values():
        group["accuracy"] = group["correct"] / group["total"]
    return {"schema_version": "3.0", "benchmark": config["dataset"]["name"], "total": len(scored_results),
            "execution_records": len(results), "non_scoring_records": len(results) - len(scored_results),
            "correct": correct, "accuracy": correct / len(scored_results) if scored_results else None, "by_category": by_category,
            "partial_evaluation": config["dataset"]["name"] == "bfcl"}


def _last_number(text: str) -> str | None:
    marker = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    values = marker or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return values[-1].replace(",", "") if values else None


def _numeric_equivalent(left: str | None, right: str | None) -> bool:
    """Compare extracted numeric answers exactly, without float rounding."""
    if left is None or right is None:
        return False
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _aime_integer(text: str) -> int | None:
    marker = re.findall(r"####\s*(\d{1,3})\b", text)
    boxed = re.findall(r"\\boxed\{\s*(\d{1,3})\s*\}", text)
    values = marker or boxed or re.findall(r"(?<![.\d])\d{1,3}(?![.\d])", text)
    return int(values[-1]) if values else None


def _math_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    marker = re.findall(r"####\s*(.+)", text)
    values = marker or boxed
    return values[-1].strip().rstrip(".") if values else None


def _math_equivalent(actual: str, expected: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    if normalize(actual) == normalize(expected):
        return True
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        return bool(sympy.simplify(parse_latex(actual) - parse_latex(expected)) == 0)
    except Exception:
        return False


def _gaia_normalize(value: str) -> str:
    value = str(value).strip()
    marker = re.findall(r"####\s*(.+)", value)
    if marker:
        value = marker[-1].strip()
    value = value.replace("$", "").replace(",", "").strip().lower()
    return re.sub(r"\s+", " ", value)


def _locomo_normalize(text: str) -> list[str]:
    try:
        from nltk.stem import PorterStemmer
        stem = PorterStemmer().stem
    except ImportError:
        # The data and runtime remain usable in minimal installations; installing
        # requirements-agent-eval.txt enables exact parity with LoCoMo's scorer.
        stem = lambda token: token

    lowered = str(text).lower().replace(",", "")
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    tokens = [token for token in without_punctuation.split() if token not in {"a", "an", "the", "and"}]
    return [stem(token) for token in tokens]


def _locomo_f1(prediction: str, ground_truth: str) -> float:
    from collections import Counter

    predicted = _locomo_normalize(prediction)
    expected = _locomo_normalize(ground_truth)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _locomo_multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [part.strip() for part in prediction.split(",")]
    answers = [part.strip() for part in ground_truth.split(",")]
    return sum(max(_locomo_f1(candidate, answer) for candidate in predictions) for answer in answers) / len(answers)


def _bfcl_submission(record: EvaluationRecord, output: AgentOutput) -> dict[str, Any]:
    result = output.metadata.get("bfcl_calls_by_turn") if record.input["task_family"] == "multiturn" else [
        [{call["name"]: call["args"]} for call in output.tool_calls]
    ]
    return {"id": record.input["id"], "result": result}


def _fallback_ast_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]], *, unordered: bool) -> bool:
    def matches(call: dict[str, Any], candidate: dict[str, Any]) -> bool:
        if len(call) != 1 or len(candidate) != 1:
            return False
        name, args = next(iter(call.items()))
        expected_name, possibilities = next(iter(candidate.items()))
        if name != expected_name or set(args) - set(possibilities):
            return False
        for key, accepted in possibilities.items():
            if key not in args:
                if "" in accepted:
                    continue
                return False
            if args[key] not in accepted:
                return False
        return True

    if len(actual) != len(expected):
        return False
    if not unordered:
        return all(matches(call, candidate) for call, candidate in zip(actual, expected))
    remaining = list(expected)
    for call in actual:
        for index, candidate in enumerate(remaining):
            if matches(call, candidate):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def _read_outputs(path: Path) -> list[AgentOutput]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(AgentOutput.from_dict(json.loads(line)))
    return values
