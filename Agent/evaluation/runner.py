from __future__ import annotations

from pathlib import Path
from typing import Any

from Agent.evaluation.config import load_run_config
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
    evaluator: Any | None = None,
    profile: str | None = None,
) -> Path:
    """Execute the single-file v2 evaluation contract."""
    del evaluator
    config = load_run_config(config_path, project_root / "Agent")
    if config.get("schema_version") != "2.0":
        raise ValueError("only single-file evaluation YAML schema_version 2.0 is supported")
    return await execute_direct_run(
        config, config_path, project_root, limit=limit, resume=resume,
        rescore=rescore, profile=profile, runtime=runtime,
    )
