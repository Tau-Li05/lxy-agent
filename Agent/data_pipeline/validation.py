from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .io import file_hash, read_jsonl


SCHEMA_FILES = {
    "canonical": "canonical-record-v1.schema.json",
    "sft": "sft-record-v1.schema.json",
    "rl_policy": "rl-policy-v1.schema.json",
    "rl_reward": "rl-reward-v1.schema.json",
    "eval_input": "eval-input-v1.schema.json",
    "eval_reference": "eval-reference-v1.schema.json",
}


def load_validators(agent_root: Path) -> dict[str, Draft202012Validator]:
    schema_dir = agent_root / "schemas"
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in schema_dir.glob("*.schema.json")
    }
    registry = Registry()
    for filename, schema in schemas.items():
        registry = registry.with_resource(filename, Resource.from_contents(schema))
    validators = {}
    for name, filename in SCHEMA_FILES.items():
        schema = schemas[filename]
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(schema, registry=registry)
    return validators


def validate_records(records: Iterable[dict[str, Any]], validator: Draft202012Validator, label: str) -> None:
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"{label}[{index}] schema error: {errors[0].message}")


def validate_jsonl(path: Path, validator: Draft202012Validator) -> list[dict[str, Any]]:
    records = [record for _, record in read_jsonl(path)]
    validate_records(records, validator, path.as_posix())
    return records


def ensure_unique_ids(records: list[dict[str, Any]], label: str) -> None:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate IDs detected")


def ensure_same_ids(left: list[dict[str, Any]], right: list[dict[str, Any]], label: str) -> None:
    if {x["id"] for x in left} != {x["id"] for x in right}:
        raise ValueError(f"{label}: input/reference ID sets differ")


def ensure_policy_isolated(records: list[dict[str, Any]]) -> None:
    forbidden = {"target", "reward", "reference", "answer", "solution", "response", "final_answer"}

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.casefold() in forbidden:
                    raise ValueError(f"RL policy contains forbidden key: {'.'.join(path + (key,))}")
                walk(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))

    for record in records:
        walk(record)


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def compare_builds(left: Path, right: Path) -> None:
    left_hashes = artifact_hashes(left)
    right_hashes = artifact_hashes(right)
    if left_hashes != right_hashes:
        left_only = sorted(set(left_hashes) ^ set(right_hashes))
        changed = sorted(key for key in set(left_hashes) & set(right_hashes) if left_hashes[key] != right_hashes[key])
        raise ValueError(f"Determinism check failed; path differences={left_only}, changed={changed}")

