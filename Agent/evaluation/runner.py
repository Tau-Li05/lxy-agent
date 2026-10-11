from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from Agent.agents.config import load_agent_config
from Agent.data_pipeline.io import file_hash, read_jsonl, write_json, write_jsonl
from Agent.loaders.competition_math import load_competition_math_dev
from Agent.reporting.dev_report import build_dev_report
from Agent.runtime.langchain_runtime import LangChainRuntime

from .adapters import BoxedLatexOutputAdapter
from .aggregation import CompetitionMathAggregator
from .checkpoints import append_checkpoint, load_checkpoints
from .config import load_run_config, resolve_path
from .math_verify_evaluator import MathVerifyEvaluator
from .models import AgentOutput
from .protocols import AgentRuntime


async def execute_run(
    config_path: Path,
    project_root: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
    rescore: bool = False,
    runtime: AgentRuntime | None = None,
    evaluator: Any | None = None,
) -> Path:
    agent_root = project_root / "Agent"
    run_config = load_run_config(config_path, agent_root)
    agent_path = resolve_path(run_config["agent_config"], config_path, project_root)
    agent_config = load_agent_config(agent_path, agent_root)
    inputs_path = resolve_path(run_config["dataset"]["inputs"], config_path, project_root)
    references_path = resolve_path(run_config["dataset"]["references"], config_path, project_root)
    output_root = resolve_path(run_config["output_root"], config_path, project_root)
    run_dir = output_root / run_config["run_id"]
    checkpoint_path = run_dir / "checkpoints" / "agent_outputs.jsonl"

    if resume and rescore:
        raise ValueError("--resume and --rescore are mutually exclusive")
    if run_dir.exists() and not (resume or rescore):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    if not run_dir.exists() and (resume or rescore):
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_competition_math_dev(inputs_path, references_path, agent_root)
    inputs = dataset.inputs[:limit] if limit is not None else dataset.inputs
    if not inputs:
        raise ValueError("run has no selected inputs")
    selected_ids = {item.id for item in inputs}

    resolved = {
        "schema_version": "1.0",
        "run": run_config,
        "agent": agent_config,
        "resolved_paths": {
            "agent_config": str(agent_path),
            "inputs": str(inputs_path),
            "references": str(references_path),
            "output": str(run_dir),
        },
        "limit": limit,
    }
    if not rescore:
        write_json(run_dir / "resolved_config.json", resolved)

    if rescore:
        outputs = _load_rescore_outputs(run_dir)
        missing = selected_ids - outputs.keys()
        if missing:
            raise ValueError(f"cannot rescore; missing {len(missing)} agent outputs")
        outputs = {key: value for key, value in outputs.items() if key in selected_ids}
    else:
        outputs = load_checkpoints(checkpoint_path) if resume else {}
        unexpected = set(outputs) - selected_ids
        if unexpected:
            raise ValueError(f"checkpoint contains IDs outside selected run: {sorted(unexpected)[:3]}")
        active_runtime = runtime or LangChainRuntime(agent_config)

        async def process(item: Any) -> AgentOutput:
            output = await active_runtime.run(item)
            if output.id != item.id:
                raise ValueError(f"runtime changed sample ID: {item.id} -> {output.id}")
            _validate_agent_output(output.to_dict(), agent_root)
            append_checkpoint(checkpoint_path, output)
            return output

        pending = [item for item in inputs if item.id not in outputs]
        completed = await asyncio.gather(*(process(item) for item in pending))
        outputs.update({output.id: output for output in completed})

    ordered_outputs = [outputs[item.id] for item in sorted(inputs, key=lambda value: value.id)]
    adapter = BoxedLatexOutputAdapter()
    predictions = [adapter.adapt(output) for output in ordered_outputs]
    active_evaluator = evaluator or MathVerifyEvaluator()
    results = [active_evaluator.evaluate(prediction, dataset.references[prediction.id]) for prediction in predictions]
    metrics = CompetitionMathAggregator().aggregate(results)
    report = build_dev_report(metrics)

    _validate_many([item.to_dict() for item in predictions], "prediction-v1.schema.json", agent_root, "predictions")
    _validate_many([item.to_dict() for item in results], "sample-result-v1.schema.json", agent_root, "sample results")
    _validate_many([metrics.to_dict()], "benchmark-metrics-v1.schema.json", agent_root, "benchmark metrics")

    write_jsonl(run_dir / "agent_outputs.jsonl", (item.to_dict() for item in ordered_outputs))
    write_jsonl(run_dir / "predictions.jsonl", (item.to_dict() for item in predictions))
    write_jsonl(run_dir / "sample_results.jsonl", (item.to_dict() for item in results))
    write_jsonl(run_dir / "errors.jsonl", (item.to_dict() for item in results if item.error is not None))
    write_json(run_dir / "dev_metrics.json", report)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_config["run_id"],
        "evaluation_scope": "development",
        "split": "validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(results),
        "inputs_sha256": file_hash(inputs_path),
        "references_sha256": file_hash(references_path),
        "agent_config_sha256": file_hash(agent_path),
        "artifacts": {
            name: file_hash(run_dir / name)
            for name in ("agent_outputs.jsonl", "predictions.jsonl", "sample_results.jsonl", "dev_metrics.json", "errors.jsonl")
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _load_rescore_outputs(run_dir: Path) -> dict[str, AgentOutput]:
    path = run_dir / "agent_outputs.jsonl"
    if not path.exists():
        path = run_dir / "checkpoints" / "agent_outputs.jsonl"
    records: dict[str, AgentOutput] = {}
    for _, value in read_jsonl(path):
        output = AgentOutput.from_dict(value)
        if output.id in records:
            raise ValueError(f"duplicate agent output ID: {output.id}")
        records[output.id] = output
    return records


def _validate_agent_output(value: dict[str, Any], agent_root: Path) -> None:
    _validate_many([value], "agent-output-v1.schema.json", agent_root, "AgentOutput")


def _validate_many(values: list[dict[str, Any]], schema_name: str, agent_root: Path, label: str) -> None:
    schema = json.loads((agent_root / "schemas" / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, value in enumerate(values):
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"{label}[{index}] schema error: {errors[0].message}")
