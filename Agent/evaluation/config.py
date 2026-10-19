from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


def load_run_config(path: Path, agent_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config must be an object")
    if value.get("schema_version") != "2.0":
        raise ValueError(f"{path}: only schema_version 2.0 is supported")
    schema = json.loads((agent_root / "schemas/run-config-v2.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{path}: {errors[0].message}")
    if value.get("schema_version") == "2.0" and value["model"]["default_profile"] not in value["model"]["profiles"]:
        raise ValueError(f"{path}: model.default_profile is not defined in model.profiles")
    for name, profile in value["model"]["profiles"].items():
        if profile["max_tokens"] >= profile["context_window"]:
            raise ValueError(f"{path}: model.profiles.{name}.max_tokens must be smaller than context_window")
    return value


def resolve_path(value: str, config_path: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("./") or value.startswith("../"):
        return (config_path.parent / path).resolve()
    return (project_root / path).resolve()
