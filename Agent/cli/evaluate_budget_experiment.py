"""Run isolated AIME budget experiments without changing the baseline runtime."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from Agent.evaluation.config import load_run_config
from Agent.evaluation.runner import execute_run
from Agent.runtime.qwen_harness_v1 import QwenHarnessRuntimeV1
from Agent.runtime.qwen_harness_v2 import QwenHarnessRuntimeV2
from Agent.runtime.qwen_harness_v3 import QwenHarnessRuntimeV3
from Agent.runtime.qwen_harness_v4 import QwenHarnessRuntimeV4
from Agent.runtime.qwen_harness_v5 import QwenHarnessRuntimeV5
from Agent.runtime.qwen_harness_v6 import QwenHarnessRuntimeV6
from Agent.runtime.qwen_harness_v7 import QwenHarnessRuntimeV7
from Agent.runtime.qwen_harness_v8 import QwenHarnessRuntimeV8
from Agent.runtime.qwen_harness_v9 import QwenHarnessRuntimeV9
from Agent.runtime.qwen_harness_v10 import QwenHarnessRuntimeV10
from Agent.runtime.qwen_harness_v11 import QwenHarnessRuntimeV11
from Agent.runtime.qwen_harness_v12 import QwenHarnessRuntimeV12
from Agent.runtime.qwen_harness_v13 import QwenHarnessRuntimeV13
from Agent.runtime.qwen_harness_v14 import QwenHarnessRuntimeV14
from Agent.runtime.qwen_harness_v15 import QwenHarnessRuntimeV15
from Agent.runtime.qwen_harness_v16 import QwenHarnessRuntimeV16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"], required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("Agent/configs/v3/aime_qwen_harness.yaml"))
    parser.add_argument("--profile", default="qwen3_5_4b")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, help="Optional smoke-test sample count")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    config = load_run_config(config_path, project_root / "Agent")
    runtime_class = {"v1": QwenHarnessRuntimeV1, "v2": QwenHarnessRuntimeV2,
                     "v3": QwenHarnessRuntimeV3, "v4": QwenHarnessRuntimeV4,
                     "v5": QwenHarnessRuntimeV5, "v6": QwenHarnessRuntimeV6,
                     "v7": QwenHarnessRuntimeV7, "v8": QwenHarnessRuntimeV8,
                     "v9": QwenHarnessRuntimeV9, "v10": QwenHarnessRuntimeV10,
                     "v11": QwenHarnessRuntimeV11, "v12": QwenHarnessRuntimeV12,
                     "v13": QwenHarnessRuntimeV13, "v14": QwenHarnessRuntimeV14,
                     "v15": QwenHarnessRuntimeV15, "v16": QwenHarnessRuntimeV16}[args.version]
    runtime = runtime_class(config, project_root, args.profile)
    output = asyncio.run(execute_run(config_path, project_root, runtime=runtime,
                                     profile=args.profile, run_name=args.run_name,
                                     limit=args.limit, resume=args.resume))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
