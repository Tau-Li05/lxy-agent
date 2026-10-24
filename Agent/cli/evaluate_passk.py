from __future__ import annotations

import argparse
import asyncio
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.evaluation.capability_metrics import pass_at_k
from Agent.evaluation.config import load_run_config, resolve_path
from Agent.evaluation.direct_dataset import load_evaluation_dataset
from Agent.evaluation.direct_runner import _runtime_for, _score
from Agent.evaluation.io import write_json, write_jsonl


def _seed(experiment_id: str, sample_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{experiment_id}:{sample_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


async def execute_passk(config_path: Path, project_root: Path, *, profile: str | None,
                        limit: int | None, trajectories: int = 8) -> Path:
    config = load_run_config(config_path, project_root / "Agent")
    dataset_path = resolve_path(config["dataset"]["path"], config_path, project_root)
    records = load_evaluation_dataset(dataset_path, project_root / "Agent", config)
    records = records[:limit] if limit is not None else records
    profile_name = profile or config["model"]["default_profile"]
    experiment_id = f"{config_path.stem}-{profile_name}-passk"
    run_dir = project_root / "runs" / experiment_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already contains results: {run_dir}")
    run_dir.mkdir(parents=True)

    trajectory_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for record in records:
        for index in range(trajectories):
            active = deepcopy(config)
            selected = active["model"]["profiles"][profile_name]
            selected["temperature"] = 0 if index == 0 else 0.7
            selected["seed"] = _seed(experiment_id, record.input["id"], index)
            runtime = _runtime_for(active, project_root, profile_name)
            runtime.run_state_root = run_dir
            output = await runtime.run(record.input)
            scored = _score(active, record, output, project_root)
            trajectory_id = output.metadata.get("trajectory_id") or f"{record.input['id']}:{index}"
            trajectory_rows.append({
                "sample_id": record.input["id"], "trajectory_id": trajectory_id,
                "trajectory_index": index, "seed": selected["seed"],
                "seed_supported": True, "temperature": selected["temperature"],
                "correct": bool(scored.get("correct")), "status": output.status,
                "usage": output.usage, "timing": output.timing,
            })
            detail_rows.append({"sample_id": record.input["id"], "trajectory_index": index,
                                "output": output.to_dict(), "score": scored})
    write_jsonl(run_dir / "trajectories.jsonl", trajectory_rows)
    write_jsonl(run_dir / "metric_details.jsonl", detail_rows)
    write_json(run_dir / "metrics.json", pass_at_k(trajectory_rows))
    write_json(run_dir / "manifest.json", {
        "schema_version": "1.0", "experiment_id": experiment_id, "config": config,
        "profile": profile_name, "trajectories_per_sample": trajectories,
        "pass_k": [1, 4, 8], "sampling_temperature": {"pass@1": 0, "others": 0.7},
    })
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent trajectories for Pass@k")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trajectories", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.trajectories < 8:
        raise SystemExit("--trajectories must be at least 8 for Pass@8")
    project_root = Path(__file__).resolve().parents[2]
    output = asyncio.run(execute_passk(args.config.resolve(), project_root, profile=args.profile,
                                       limit=args.limit, trajectories=args.trajectories))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
