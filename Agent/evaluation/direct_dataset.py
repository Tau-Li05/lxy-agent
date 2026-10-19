from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class EvaluationRecord:
    input: dict[str, Any]
    reference: dict[str, Any] | None


def load_evaluation_dataset(path: Path, agent_root: Path, config: dict[str, Any]) -> list[EvaluationRecord]:
    schema = json.loads((agent_root / "schemas/evaluation-record-v2.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    records: list[EvaluationRecord] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
            if errors:
                raise ValueError(f"{path}:{line_number}: {errors[0].message}")
            item = value["input"]
            _validate_semantics(item, config, path, line_number)
            if item["id"] in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {item['id']}")
            ids.add(item["id"])
            records.append(EvaluationRecord(input=item, reference=value["reference"]))
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def _validate_semantics(item: dict[str, Any], config: dict[str, Any], path: Path, line: int) -> None:
    prefix = f"{path}:{line}"
    if item["task_family"] != config["task_family"]:
        raise ValueError(f"{prefix}: task_family does not match YAML")
    if item["metadata"].get("benchmark") != config["dataset"]["name"]:
        raise ValueError(f"{prefix}: benchmark does not match dataset.name")
    if item["task_family"] == "single_turn" and len(item["turns"]) != 1:
        raise ValueError(f"{prefix}: single_turn requires exactly one turn")
    if config["dataset"]["name"] in {"gsm8k", "aime"}:
        if item["tools"] or config["tool"] != {"enabled": False, "provider": "none", "schema_source": "none", "execution": "disabled"}:
            raise ValueError(f"{prefix}: math single-turn datasets must disable tools")
    elif config["dataset"]["name"] == "locomo":
        questions = item["environment"].get("questions")
        if (item["task_family"] != "multiturn" or not item["tools"] or not questions
                or item["environment"].get("memory_backend") != "sqlite_fts5"
                or config["tool"] != {"enabled": True, "provider": "locomo_memory",
                                      "schema_source": "dataset", "execution": "enabled"}):
            raise ValueError(f"{prefix}: LoCoMo requires session turns, sampled questions, and SQLite FTS5 tools")
    elif item["task_family"] == "single_turn":
        is_irrelevance = "irrelevance" in item["metadata"].get("category", "")
        if (not item["tools"] and not is_irrelevance) or config["tool"]["execution"] != "disabled":
            raise ValueError(f"{prefix}: BFCL single-turn requires schemas and disabled execution")
    else:
        required = {"initial_config", "involved_classes"}
        if config["tool"]["provider"] != "bfcl" or config["tool"]["execution"] != "enabled" or not required <= item["environment"].keys():
            raise ValueError(f"{prefix}: BFCL multiturn requires its backend and initial state")
