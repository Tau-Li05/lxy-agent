from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from Agent.data_pipeline.io import read_jsonl
from Agent.data_pipeline.validation import ensure_same_ids, ensure_unique_ids, load_validators, validate_records
from Agent.evaluation.models import AgentInput, Reference


@dataclass(frozen=True)
class DevelopmentDataset:
    inputs: list[AgentInput]
    references: dict[str, Reference]


def load_competition_math_dev(inputs_path: Path, references_path: Path, agent_root: Path) -> DevelopmentDataset:
    validators = load_validators(agent_root)
    raw_inputs = [row for _, row in read_jsonl(inputs_path)]
    raw_references = [row for _, row in read_jsonl(references_path)]
    validate_records(raw_inputs, validators["eval_input"], "DEV inputs")
    validate_records(raw_references, validators["eval_reference"], "DEV references")
    ensure_unique_ids(raw_inputs, "DEV inputs")
    ensure_unique_ids(raw_references, "DEV references")
    ensure_same_ids(raw_inputs, raw_references, "competition_math DEV")

    for record in raw_inputs + raw_references:
        if record["evaluation_scope"] != "development" or record["metadata"]["split"] != "validation":
            raise ValueError("competition_math DEV requires evaluation_scope=development and split=validation")
        if record["metadata"].get("benchmark") != "competition_math":
            raise ValueError("competition_math DEV contains another benchmark")

    agent_input_validator = _load_schema_validator(agent_root / "schemas" / "agent-input-v1.schema.json")
    inputs: list[AgentInput] = []
    for record in raw_inputs:
        value = {"id": record["id"], **record["input"], "metadata": record["metadata"]}
        _validate_one(agent_input_validator, value, record["id"])
        inputs.append(AgentInput(**value))

    input_metadata = {record["id"]: record["metadata"] for record in raw_inputs}
    references = {
        record["id"]: Reference(
            id=record["id"],
            reference=record["reward"]["reference"],
            answer_extractor=record["reward"]["answer_extractor"],
            evaluator_version=record["reward"]["evaluator_version"],
            normalization=list(record["reward"].get("normalization", [])),
            metadata={**record["metadata"], **input_metadata[record["id"]]},
        )
        for record in raw_references
    }
    return DevelopmentDataset(inputs=inputs, references=references)


def _load_schema_validator(path: Path) -> Draft202012Validator:
    import json

    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_one(validator: Draft202012Validator, value: dict, label: str) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{label}: AgentInput schema error: {errors[0].message}")
