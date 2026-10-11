from __future__ import annotations

import argparse
from pathlib import Path

from .builder import run_build


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m Agent.data_pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build deterministic canonical and training/evaluation views")
    choice = build.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true", help="Build every v1 dataset")
    choice.add_argument("--dataset", help="Build one dataset for development validation only")
    build.add_argument("--output-root", default="data")
    build.add_argument("--verify-only", action="store_true")
    build.add_argument("--publish", action="store_true")
    build.add_argument("--force-rebuild", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build":
        if args.verify_only and args.publish:
            raise SystemExit("--verify-only and --publish are mutually exclusive")
        workspace_root = Path.cwd().resolve()
        agent_root = Path(__file__).resolve().parents[1]
        build_id, location = run_build(
            agent_root, workspace_root, (workspace_root / args.output_root).resolve(),
            args.dataset, args.verify_only, args.publish, args.force_rebuild,
        )
        mode = "published" if args.publish else "verified/staged"
        print(f"Build {build_id} {mode}: {location}")
        return 0
    return 2

