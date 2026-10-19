from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from Agent.evaluation.runner import execute_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Agent evaluation YAML")
    parser.add_argument("--config", type=Path, required=True, help="evaluation run YAML")
    parser.add_argument("--limit", type=int, help="evaluate only the first N records")
    parser.add_argument("--profile", help="override model.default_profile")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue an existing run")
    mode.add_argument("--rescore", action="store_true", help="reuse AgentOutput and rerun scoring")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    project_root = Path(__file__).resolve().parents[2]
    output = asyncio.run(
        execute_run(
            args.config.resolve(),
            project_root,
            limit=args.limit,
            resume=args.resume,
            rescore=args.rescore,
            profile=args.profile,
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
