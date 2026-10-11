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
    schema = json.loads((agent_root / "schemas" / "evaluation-run-config-v1.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{path}: {errors[0].message}")
    return value


def resolve_path(value: str, config_path: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("./") or value.startswith("../"):
        return (config_path.parent / path).resolve()
    return (project_root / path).resolve()
