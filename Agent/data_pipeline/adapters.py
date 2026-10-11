from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .extractors import EXTRACTORS
from .io import object_hash, read_jsonl
from .splitting import stratified_split


@dataclass
class ConversionResult:
    benchmark: str
    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    source_files: list[Path] = field(default_factory=list)
    split_policy: dict[str, Any] = field(default_factory=dict)


def normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _source(
    config: dict[str, Any], raw: dict[str, Any], path: Path, row_index: int, source_id: Any, source_split: str, root: Path
) -> tuple[dict[str, Any], str]:
    record_hash = object_hash(raw)
    source = {
        "dataset": config["benchmark"],
        "dataset_version": config["dataset_version"],
        "source_split": source_split,
        "source_id": source_id,
        "source_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "row_index": row_index,
        "record_hash": record_hash,
        "license": config.get("license"),
    }
    return source, record_hash


def _canonical(
    config: dict[str, Any], raw: dict[str, Any], path: Path, row_index: int, source_split: str,
    split: str, source_id: Any, messages: list[dict[str, str]], response: str | None,
    final_answer: Any, reward: dict[str, Any], metadata: dict[str, Any], root: Path,
) -> dict[str, Any]:
    source, record_hash = _source(config, raw, path, row_index, source_id, source_split, root)
    id_part = str(source_id) if source_id is not None else record_hash
    return {
        "schema_version": "1.0",
        "id": f"{config['benchmark']}:{split}:{id_part}",
        "benchmark": config["benchmark"],
        "source": source,
        "split": split,
        "usage": config["usage_by_split"][split],
        "task_type": config["task_type"],
        "input": {"messages": messages, "tools": [], "attachments": [], "environment": {}},
        "target": {"response": response, "final_answer": final_answer},
        "reward": reward,
        "metadata": metadata,
    }


def _reject(config: dict[str, Any], path: Path, row_index: int, raw: dict[str, Any], reason: str, root: Path) -> dict[str, Any]:
    return {
        "benchmark": config["benchmark"],
        "reason": reason,
        "source_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "row_index": row_index,
        "record_hash": object_hash(raw),
    }


def competition_math(config: dict[str, Any], config_path: Path, root: Path) -> ConversionResult:
    path = (config_path.parent / config["sources"][0]["path"]).resolve()
    result = ConversionResult(config["benchmark"], source_files=[path])
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in read_jsonl(path):
        question = raw.get("problem")
        solution = raw.get("solution")
        level = raw.get("level")
        category = raw.get("type")
        if not isinstance(question, str) or not question.strip():
            result.rejected.append(_reject(config, path, row_index, raw, "empty_question", root)); continue
        if not isinstance(solution, str) or not solution.strip():
            result.rejected.append(_reject(config, path, row_index, raw, "empty_solution", root)); continue
        if level not in {f"Level {index}" for index in range(1, 6)}:
            result.rejected.append(_reject(config, path, row_index, raw, "invalid_level", root)); continue
        answer = EXTRACTORS["boxed_latex"](solution)
        if not answer:
            result.rejected.append(_reject(config, path, row_index, raw, "answer_extraction_failed", root)); continue
        question_key = normalize_question(question)
        if question_key in seen:
            result.rejected.append(_reject(config, path, row_index, raw, "duplicate_question", root)); continue
        seen.add(question_key)
        candidates.append({"raw": raw, "row_index": row_index, "answer": answer, "question_key": question_key})
    policy = config["split_policy"]
    split_records = stratified_split(
        candidates,
        stratum=lambda item: (item["raw"]["level"], item["raw"]["type"]),
        key=lambda item: object_hash(item["raw"]),
        seed=policy["seed"],
        validation_count=policy["counts"]["validation"],
        test_count=policy["counts"]["test"],
    )
    prompt = config["prompt_template"]
    for split, items in split_records.items():
        for item in items:
            raw = item["raw"]
            result.records.append(_canonical(
                config, raw, path, item["row_index"], "unspecified", split, None,
                [{"role": "user", "content": prompt.replace("{{ question }}", raw["problem"])}],
                raw["solution"], item["answer"],
                {"type": "math_equivalence", "reference": item["answer"], "normalization": ["strip", "normalize_latex"], "answer_extractor": "boxed_latex", "evaluator_version": "1", "parameters": {}},
                {"level": raw["level"], "category": raw["type"]}, root,
            ))
    result.records.sort(key=lambda record: record["id"])
    result.split_policy = policy
    return result


def math_jsonl(config: dict[str, Any], config_path: Path, root: Path) -> ConversionResult:
    result = ConversionResult(config["benchmark"])
    prompt = config["prompt_template"]
    extractor = EXTRACTORS[config["answer_extractor"]]
    for source_config in config["sources"]:
        path = (config_path.parent / source_config["path"]).resolve()
        result.source_files.append(path)
        split = source_config["split"]
        source_split = source_config.get("source_split", split)
        for row_index, raw in read_jsonl(path):
            question = raw.get(config["fields"]["question"])
            raw_answer = raw.get(config["fields"]["answer"])
            if not isinstance(question, str) or not question.strip():
                result.rejected.append(_reject(config, path, row_index, raw, "empty_question", root)); continue
            answer = extractor(raw_answer)
            if answer is None or answer == "":
                result.rejected.append(_reject(config, path, row_index, raw, "answer_extraction_failed", root)); continue
            source_id = raw.get(config["fields"].get("id", "")) if config["fields"].get("id") else None
            response_field = config["fields"].get("response")
            response = raw.get(response_field) if response_field else None
            metadata = {target: raw[source] for target, source in config.get("metadata_fields", {}).items() if source in raw}
            result.records.append(_canonical(
                config, raw, path, row_index, source_split, split, source_id,
                [{"role": "user", "content": prompt.replace("{{ question }}", question)}],
                response, answer,
                {"type": config["reward_type"], "reference": answer, "normalization": config["normalization"], "answer_extractor": config["output_extractor"], "evaluator_version": "1", "parameters": {}},
                metadata, root,
            ))
    result.records.sort(key=lambda record: record["id"])
    result.split_policy = {"type": "upstream"}
    return result


def livebench_math(config: dict[str, Any], config_path: Path, root: Path) -> ConversionResult:
    source_config = config["sources"][0]
    path = (config_path.parent / source_config["path"]).resolve()
    result = ConversionResult(config["benchmark"], source_files=[path], split_policy={"type": "upstream"})
    for row_index, raw in read_jsonl(path):
        turns = raw.get("turns")
        answer = EXTRACTORS["livebench_answer"](raw.get("ground_truth"))
        if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], str):
            raise ValueError(f"{path}:{row_index + 1}: LiveBench v1 requires exactly one string turn")
        if answer is None:
            result.rejected.append(_reject(config, path, row_index, raw, "answer_extraction_failed", root)); continue
        metadata = {key: raw.get(key) for key in config["metadata_fields"]}
        result.records.append(_canonical(
            config, raw, path, row_index, "test", "test", raw["question_id"],
            [{"role": "user", "content": turns[0]}], None, answer,
            {"type": "exact_match", "reference": answer, "normalization": ["strip"], "answer_extractor": "answer_label", "evaluator_version": "1", "parameters": {}},
            metadata, root,
        ))
    result.records.sort(key=lambda record: record["id"])
    return result


ADAPTERS = {
    "competition_math": competition_math,
    "math_jsonl": math_jsonl,
    "livebench_math": livebench_math,
}

