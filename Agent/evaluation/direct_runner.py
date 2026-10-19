from __future__ import annotations

import asyncio
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

from Agent.evaluation.checkpoints import append_checkpoint, load_checkpoints
from Agent.evaluation.config import resolve_path
from Agent.evaluation.direct_dataset import EvaluationRecord, load_evaluation_dataset
from Agent.evaluation.io import write_json, write_jsonl
from Agent.evaluation.models import AgentOutput
from Agent.runtime.direct_runtime import DirectLangChainRuntime


async def execute_direct_run(
    config: dict[str, Any], config_path: Path, project_root: Path, *, limit: int | None = None,
    resume: bool = False, rescore: bool = False, profile: str | None = None, runtime: Any | None = None,
) -> Path:
    dataset_path = resolve_path(config["dataset"]["path"], config_path, project_root)
    records = load_evaluation_dataset(dataset_path, project_root / "Agent", config)
    records = records[:limit] if limit is not None else records
    profile_name = profile or config["model"]["default_profile"]
    if profile_name not in config["model"]["profiles"]:
        raise ValueError(f"unknown model profile: {profile_name}")
    run_dir = project_root / "runs" / f"{config_path.stem}-{profile_name}"
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
        active_runtime = runtime or DirectLangChainRuntime(config, project_root, profile_name)
        if hasattr(active_runtime, "run_state_root"):
            active_runtime.run_state_root = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(4)

        async def process(record: EvaluationRecord) -> AgentOutput:
            async with semaphore:
                output = await active_runtime.run(record.input)
                append_checkpoint(checkpoint, output)
                return output

        pending = [record for record in records if record.input["id"] not in outputs]
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
    if config["dataset"]["name"] == "bfcl":
        write_jsonl(run_dir / "bfcl_results.jsonl", (_bfcl_submission(record, outputs[record.input["id"]]) for record in records))
    return run_dir


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
        correct = predicted is not None and predicted == expected
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct), "correct": correct,
                "prediction": predicted, "reference": expected}
    if config["dataset"]["name"] == "aime":
        predicted = _aime_integer(output.final_answer or "")
        expected = int((record.reference or {})["answer"])
        correct = predicted == expected
        return {"id": output.id, "status": "correct" if correct else "incorrect", "score": float(correct),
                "correct": correct, "prediction": predicted, "reference": expected,
                "metadata": {"category": category}}
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
        if record.input["task_family"] == "multiturn":
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
        return {"schema_version": "2.0", "benchmark": "locomo", "total": len(questions),
                "score_sum": total_score, "mean_f1": total_score / len(questions),
                "by_category": by_category, "conversation_count": len(results),
                "sampling_rate": 0.10}
    correct = sum(bool(result["correct"]) for result in results)
    by_category: dict[str, dict[str, int | float]] = {}
    for result in results:
        category = result.get("metadata", {}).get("category", "main")
        group = by_category.setdefault(category, {"total": 0, "correct": 0, "accuracy": 0.0})
        group["total"] += 1
        group["correct"] += int(result["correct"])
    for group in by_category.values():
        group["accuracy"] = group["correct"] / group["total"]
    return {"schema_version": "2.0", "benchmark": config["dataset"]["name"], "total": len(results),
            "correct": correct, "accuracy": correct / len(results), "by_category": by_category,
            "partial_evaluation": config["dataset"]["name"] == "bfcl"}


def _last_number(text: str) -> str | None:
    marker = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    values = marker or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return values[-1].replace(",", "") if values else None


def _aime_integer(text: str) -> int | None:
    marker = re.findall(r"####\s*(\d{1,3})\b", text)
    boxed = re.findall(r"\\boxed\{\s*(\d{1,3})\s*\}", text)
    values = marker or boxed or re.findall(r"(?<![.\d])\d{1,3}(?![.\d])", text)
    return int(values[-1]) if values else None


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
