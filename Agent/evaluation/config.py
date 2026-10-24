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
    if value.get("schema_version") != "3.0":
        raise ValueError(f"{path}: only schema_version 3.0 is supported")
    schema = json.loads((agent_root / "schemas/run-config-v3.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{path}: {errors[0].message}")
    if value["model"]["default_profile"] not in value["model"]["profiles"]:
        raise ValueError(f"{path}: model.default_profile is not defined in model.profiles")
    for name, profile in value["model"]["profiles"].items():
        if profile["max_tokens"] >= profile["context_window"]:
            raise ValueError(f"{path}: model.profiles.{name}.max_tokens must be smaller than context_window")
        provider = profile.get("provider", "openai_compatible")
        if provider == "ollama" and not (profile.get("host") or profile.get("host_env")):
            raise ValueError(f"{path}: model.profiles.{name} Ollama profile requires host or host_env")
        if provider == "openai_compatible" and not (profile.get("base_url_env") and profile.get("api_key_env")):
            raise ValueError(f"{path}: model.profiles.{name} requires base_url_env and api_key_env")
    _validate_v3_semantics(value, path)
    return value


def _validate_v3_semantics(value: dict[str, Any], path: Path) -> None:
    features = value["features"]
    tools = features.get("tools", {"enabled": False, "execution": False})
    if tools["execution"] and not tools["enabled"]:
        raise ValueError(f"{path}: features.tools.execution requires features.tools.enabled")
    for name in ("web_search", "memory"):
        if features.get(name, {}).get("enabled") and not tools["enabled"]:
            raise ValueError(f"{path}: features.{name}.enabled requires features.tools.enabled")
    kind = value["runtime"]["kind"]
    compaction = features.get("compaction", {})
    if compaction.get("enabled") and float(compaction.get("target_ratio", 0.65)) >= float(compaction.get("trigger_ratio", 0.80)):
        raise ValueError(f"{path}: features.compaction.target_ratio must be smaller than trigger_ratio")
    if kind == "llm_direct" and (tools["execution"] or any(
        features.get(name, {}).get("enabled") for name in (
            "planning", "verification", "compaction", "self_correction", "difficulty_router"
        )
    )):
        raise ValueError(f"{path}: llm_direct cannot execute tools or enable agent enhancements")
    if kind == "bfcl_protocol" and any(
        features.get(name, {}).get("enabled")
        for name in ("planning", "verification", "compaction", "self_correction", "output_recovery",
                     "finalization", "difficulty_router")
    ):
        raise ValueError(f"{path}: bfcl_protocol permits only the minimal required protocol loop")


def resolve_path(value: str, config_path: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("./") or value.startswith("../"):
        return (config_path.parent / path).resolve()
    return (project_root / path).resolve()
