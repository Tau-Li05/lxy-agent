from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Agent.evaluation.capability_metrics import paired_gain


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate1(run_dirs: list[Path]) -> dict[str, Any]:
    tagged: list[tuple[str, dict[str, Any]]] = []
    for directory in run_dirs:
        resolved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
        dataset = resolved["config"]["dataset"]["name"]
        tagged.extend((dataset, row) for row in _jsonl(directory / "agent_outputs.jsonl"))
    outputs = [row for _, row in tagged]
    total = len(outputs)
    normal = sum(row.get("status") == "success" for row in outputs)
    timeouts = sum(row.get("status") == "timeout" for row in outputs)
    bfcl = [row for dataset, row in tagged if dataset == "bfcl"]
    valid_calls = sum(bool(row.get("tool_calls")) and all(
        call.get("name") and isinstance(call.get("args"), dict) for call in row["tool_calls"])
                      for row in bfcl)
    format_rate = valid_calls / len(bfcl) if bfcl else None
    normal_rate = normal / total if total else 0.0
    timeout_rate = timeouts / total if total else 0.0
    counts = {
        "gsm8k": sum(dataset == "gsm8k" for dataset, _ in tagged),
        "aime": sum(dataset == "aime" for dataset, _ in tagged),
        "bfcl_single": len(bfcl),
    }
    expected = {"gsm8k": 50, "aime": 15, "bfcl_single": 100}
    coverage_ok = all(counts[key] >= value for key, value in expected.items())
    passed = coverage_ok and normal_rate >= 0.95 and timeout_rate <= 0.05 and format_rate is not None and format_rate >= 0.95
    return {"passed": passed, "samples": total, "normal_exit_rate": normal_rate,
            "timeout_rate": timeout_rate, "tool_call_format_rate": format_rate,
            "counts": counts, "minimum_counts": expected,
            "thresholds": {"normal_exit_rate": 0.95, "timeout_rate_max": 0.05,
                           "tool_call_format_rate": 0.95}}


def gate2(direct_run: Path, harness_run: Path) -> dict[str, Any]:
    before_rows = _jsonl(direct_run / "sample_results.jsonl")
    after_rows = _jsonl(harness_run / "sample_results.jsonl")
    before = {row["id"]: bool(row.get("correct")) for row in before_rows}
    after = {row["id"]: bool(row.get("correct")) for row in after_rows}
    gain = paired_gain(before, after)
    shared = before.keys() & after.keys()
    regressions = sum(before[key] and not after[key] for key in shared)
    improvements = sum(not before[key] and after[key] for key in shared)
    regression_rate = regressions / sum(before.values()) if sum(before.values()) else 0.0
    gain.update({"improvements": improvements, "regressions": regressions,
                 "correction_regression_rate": regression_rate,
                 "passed": bool(gain.get("gain") is not None and gain["gain"] > 0 and improvements > regressions)})
    return gain
