from __future__ import annotations

import argparse
import json
from pathlib import Path

from Agent.evaluation.gates import gate1, gate2


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen 4B rollout gates")
    sub = parser.add_subparsers(dest="gate", required=True)
    one = sub.add_parser("gate1")
    one.add_argument("runs", type=Path, nargs="+")
    two = sub.add_parser("gate2")
    two.add_argument("direct", type=Path)
    two.add_argument("harness", type=Path)
    args = parser.parse_args()
    result = gate1(args.runs) if args.gate == "gate1" else gate2(args.direct, args.harness)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
