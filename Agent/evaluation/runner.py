from __future__ import annotations

from pathlib import Path
from typing import Any

from Agent.evaluation.config import _validate_v3_semantics, load_run_config
from Agent.evaluation.direct_runner import execute_direct_run
from Agent.evaluation.protocols import AgentRuntime


async def execute_run(
    config_path: Path,
    project_root: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
    rescore: bool = False,
    runtime: AgentRuntime | None = None,
    profile: str | None = None,
    feature_overrides: dict[str, str] | None = None,
    run_name: str | None = None,
) -> Path:
    """Execute a schema-version 3 evaluation run."""
    config = load_run_config(config_path, project_root / "Agent")
    if feature_overrides:
        _apply_feature_overrides(config, feature_overrides, profile_name=profile or config["model"]["default_profile"])
        _validate_v3_semantics(config, config_path)
    return await execute_direct_run(
        config, config_path, project_root, limit=limit, resume=resume,
        rescore=rescore, profile=profile, runtime=runtime,
        run_name=run_name,
    )


def _apply_feature_overrides(config: dict[str, Any], overrides: dict[str, str], *, profile_name: str) -> None:
    if "math_tools" in overrides:
        mode = overrides["math_tools"]
        config["features"]["math_tools"] = {"mode": mode}
        enabled = mode != "none"
        config["features"]["tools"] = {"enabled": enabled, "execution": enabled}
        config["tool"] = ({"enabled": True, "provider": "math", "schema_source": "registry", "execution": "enabled"}
                          if enabled else {"enabled": False, "provider": "none", "schema_source": "none", "execution": "disabled"})
    if "verification" in overrides:
        method = overrides["verification"]
        config["features"]["verification"] = ({"enabled": False} if method == "none"
                                                  else {"enabled": True, "method": method})
        if method in {"python", "python_sympy"} and "math_tools" not in overrides:
            config["features"]["math_tools"] = {"mode": method}
            config["features"]["tools"] = {"enabled": True, "execution": True}
            config["tool"] = {"enabled": True, "provider": "math", "schema_source": "registry", "execution": "enabled"}
    for name in ("planning", "compaction"):
        if name in overrides:
            config["features"][name] = {"enabled": overrides[name] == "on"}
    if "context_window" in overrides:
        value = int(overrides["context_window"])
        profile = config["model"]["profiles"][profile_name]
        if profile["max_tokens"] >= value:
            profile["max_tokens"] = max(1, value // 4)
        profile["context_window"] = value
        config["limits"]["context_window_override"] = value
