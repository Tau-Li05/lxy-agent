#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agent.evaluation.config import load_run_config, resolve_path
from Agent.evaluation.direct_dataset import load_evaluation_dataset

AGENT_ROOT = ROOT / "Agent"


def main() -> int:
    errors: list[str] = []
    checked = 0
    for path in sorted((AGENT_ROOT / "configs/evaluations").glob("*.yaml")):
        try:
            config = load_run_config(path, AGENT_ROOT)
            if config.get("schema_version") != "2.0":
                continue
            dataset = resolve_path(config["dataset"]["path"], path, ROOT)
            load_evaluation_dataset(dataset, AGENT_ROOT, config)
            checked += 1
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        print("配置校验失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(json.dumps({"valid_v2_configs": checked}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
