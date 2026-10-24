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
    parser.add_argument("--run-name", help="explicit run directory name (required to keep multiple ablations)")
    parser.add_argument("--math-tools", choices=["none", "python", "python_sympy"], help="math-tool ablation")
    parser.add_argument("--verification", choices=["none", "self", "python", "python_sympy"], help="verification ablation")
    parser.add_argument("--planning", choices=["on", "off"], help="planning feature override")
    parser.add_argument("--compaction", choices=["on", "off"], help="compaction feature override")
    parser.add_argument("--context-window", type=int, choices=[4096, 8192, 16384, 32768, 65536], help="context robustness budget")
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
            run_name=args.run_name,
            feature_overrides={key: value for key, value in {
                "math_tools": args.math_tools, "verification": args.verification,
                "planning": args.planning, "compaction": args.compaction,
                "context_window": str(args.context_window) if args.context_window else None,
            }.items() if value is not None},
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
