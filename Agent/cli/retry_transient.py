from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from Agent.evaluation.checkpoints import append_checkpoint, load_checkpoints
from Agent.evaluation.config import load_run_config, resolve_path
from Agent.evaluation.direct_dataset import EvaluationRecord, load_evaluation_dataset
from Agent.evaluation.direct_runner import _aggregate, _bfcl_submission, _runtime_for, _score
from Agent.evaluation.io import write_json, write_jsonl
from Agent.evaluation.models import AgentOutput


logger = logging.getLogger(__name__)


def is_transient_output(output: AgentOutput) -> bool:
    if output.status == "timeout":
        return True
    error = output.error or {}
    error_type = str(error.get("type", ""))
    message = str(error.get("message", "")).lower()
    if error_type in {
        "OpenAIConnectionError", "APIConnectionError", "ConnectError",
        "ConnectionError", "ReadError", "RemoteProtocolError",
    }:
        return True
    return "502 bad gateway" in message or "internal server error" in message


def _read_outputs(path: Path) -> dict[str, AgentOutput]:
    outputs: dict[str, AgentOutput] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                output = AgentOutput.from_dict(json.loads(line))
                outputs[output.id] = output
    return outputs


def _source_outputs(run_dir: Path) -> Path:
    final = run_dir / "agent_outputs.jsonl"
    checkpoint = run_dir / "checkpoints/agent_outputs.jsonl"
    if final.exists():
        return final
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(f"no AgentOutput file found in {run_dir}")


async def retry_run(
    source_dir: Path, output_root: Path, project_root: Path, *, resume: bool, dry_run: bool,
) -> dict[str, Any]:
    resolved_path = source_dir / "resolved_config.json"
    resolved = _load_resolved_config(source_dir, resolved_path, project_root)
    config = resolved["config"]
    profile = resolved["profile"]
    dataset_path = Path(resolved["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = resolve_path(dataset_path, resolved_path, project_root)
    records = load_evaluation_dataset(dataset_path, project_root / "Agent", config)
    records_by_id = {record.input["id"]: record for record in records}
    original = _read_outputs(_source_outputs(source_dir))
    missing = records_by_id.keys() - original.keys()
    available_records = [record for record in records if record.input["id"] in original]
    targets = [record for record in available_records if is_transient_output(original[record.input["id"]])]
    summary = {
        "run": source_dir.name,
        "dataset_records": len(records),
        "source_records": len(available_records),
        "missing_records": len(missing),
        "partial_source": bool(missing),
        "retryable_records": len(targets),
        "retryable_ids": [record.input["id"] for record in targets],
    }
    if dry_run or not targets:
        return summary

    destination = output_root / source_dir.name
    retry_checkpoint = destination / "checkpoints/retried_outputs.jsonl"
    if destination.exists() and not resume:
        raise FileExistsError(f"retry output already exists: {destination}; use --resume")
    destination.mkdir(parents=True, exist_ok=True)
    retried = load_checkpoints(retry_checkpoint) if resume and retry_checkpoint.exists() else {}
    runtime = _runtime_for(config, project_root, profile)
    runtime.run_state_root = destination
    semaphore = asyncio.Semaphore(4)

    async def process(record: EvaluationRecord) -> AgentOutput:
        async with semaphore:
            logger.info("Retrying %s/%s", source_dir.name, record.input["id"])
            output = await runtime.run(record.input)
            append_checkpoint(retry_checkpoint, output)
            return output

    pending = [record for record in targets if record.input["id"] not in retried]
    completed = await asyncio.gather(*(process(record) for record in pending))
    retried.update({output.id: output for output in completed})
    if len(retried) != len(targets):
        raise ValueError(f"{source_dir}: retry checkpoint has {len(retried)} results, expected {len(targets)}")

    merged = dict(original)
    merged.update(retried)
    ordered = [merged[record.input["id"]] for record in available_records]
    results = [_score(config, record, merged[record.input["id"]], project_root) for record in available_records]
    audit = [
        {
            "id": record.input["id"],
            "script_rerun_count": 1,
            "original_attempt_count": original[record.input["id"]].metadata.get("attempt_count", 1),
            "retry_attempt_count": retried[record.input["id"]].metadata.get("attempt_count", 1),
            "total_attempt_count": (
                original[record.input["id"]].metadata.get("attempt_count", 1)
                + retried[record.input["id"]].metadata.get("attempt_count", 1)
            ),
            "before": original[record.input["id"]].to_dict(),
            "after": retried[record.input["id"]].to_dict(),
        }
        for record in targets
    ]
    write_jsonl(destination / "agent_outputs.jsonl", (output.to_dict() for output in ordered))
    write_jsonl(destination / "sample_results.jsonl", results)
    write_jsonl(destination / "retry_audit.jsonl", audit)
    write_json(destination / "metrics.json", _aggregate(config, results))
    write_json(destination / "resolved_config.json", {
        **resolved,
        "retry_source": str(source_dir.resolve()),
        "retry_policy": "timeout-or-transient-connection/server-error; two runtime retries",
    })
    if config["dataset"]["name"] == "bfcl":
        write_jsonl(destination / "bfcl_results.jsonl", (
            _bfcl_submission(record, merged[record.input["id"]]) for record in available_records
        ))
    summary["output"] = str(destination)
    summary["recovered"] = sum(
        original[sample_id].status != "success" and retried[sample_id].status == "success"
        for sample_id in retried
    )
    summary["retry_attempts"] = sum(output.metadata.get("attempt_count", 1) for output in retried.values())
    return summary


def _load_resolved_config(source_dir: Path, resolved_path: Path, project_root: Path) -> dict[str, Any]:
    if resolved_path.exists():
        return json.loads(resolved_path.read_text(encoding="utf-8"))
    config_root = project_root / "Agent/configs/v3"
    for config_path in sorted(config_root.glob("*.yaml")):
        config = load_run_config(config_path, project_root / "Agent")
        for profile in config["model"]["profiles"]:
            if source_dir.name == f"{config_path.stem}-{profile}":
                return {
                    "config": config,
                    "profile": profile,
                    "dataset_path": str(resolve_path(config["dataset"]["path"], config_path, project_root)),
                }
    raise FileNotFoundError(f"missing resolved config and cannot infer run configuration: {source_dir}")


async def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    selected = set(args.run or [])
    run_dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir() and (not selected or path.name in selected)]
    if selected - {path.name for path in run_dirs}:
        raise ValueError(f"unknown runs: {sorted(selected - {path.name for path in run_dirs})}")
    return [
        await retry_run(run_dir, output_root, project_root, resume=args.resume, dry_run=args.dry_run)
        for run_dir in run_dirs
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry transient failures from copied evaluation runs")
    parser.add_argument("--source-root", type=Path, default=Path("runs copy"))
    parser.add_argument("--output-root", type=Path, default=Path("runs_retry"))
    parser.add_argument("--run", action="append", help="retry only this run directory; repeatable")
    parser.add_argument("--resume", action="store_true", help="resume interrupted retry checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="list retryable samples without model calls")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summaries = asyncio.run(execute(build_parser().parse_args()))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
