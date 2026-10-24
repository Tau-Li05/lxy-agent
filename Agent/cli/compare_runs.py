from __future__ import annotations

import argparse
import json
from pathlib import Path

from Agent.evaluation.capability_metrics import paired_gain


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_rows(run: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in (run / "sample_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def _paired_breakdown(left: dict[str, dict], right: dict[str, dict]) -> dict:
    shared = sorted(left.keys() & right.keys())
    transitions = {
        "correct_to_correct": 0, "incorrect_to_correct": 0,
        "correct_to_incorrect": 0, "incorrect_to_incorrect": 0,
    }
    regression_reasons: dict[str, int] = {}
    for sample_id in shared:
        before, after = bool(left[sample_id].get("correct")), bool(right[sample_id].get("correct"))
        key = ("correct" if before else "incorrect") + "_to_" + ("correct" if after else "incorrect")
        transitions[key] += 1
        if before and not after:
            status = str(right[sample_id].get("status") or "unknown")
            error_type = (right[sample_id].get("error") or {}).get("type")
            reason = str(error_type or status)
            regression_reasons[reason] = regression_reasons.get(reason, 0) + 1
    return {
        "transitions": transitions,
        "corrections": transitions["incorrect_to_correct"],
        "regressions": transitions["correct_to_incorrect"],
        "net_correct_delta": transitions["incorrect_to_correct"] - transitions["correct_to_incorrect"],
        "regression_reasons": regression_reasons,
    }


def compare(before: Path, after: Path, metric: str) -> dict:
    left_manifest, right_manifest = _load(before / "manifest.json"), _load(after / "manifest.json")
    mismatches = []
    same_model_required = metric not in {"MODEL_DELTA", "FINAL_DELTA"}
    if same_model_required and left_manifest.get("exact_model_name", left_manifest.get("model", {}).get("name")) != right_manifest.get("exact_model_name", right_manifest.get("model", {}).get("name")):
        mismatches.append("exact_model_name")
    if left_manifest.get("resource_budget") != right_manifest.get("resource_budget"):
        mismatches.append("resource_budget")
    if left_manifest.get("dataset_sha256") != right_manifest.get("dataset_sha256"):
        mismatches.append("dataset_sha256")
    left_prompt = (left_manifest.get("resolved_config") or {}).get("prompt")
    right_prompt = (right_manifest.get("resolved_config") or {}).get("prompt")
    if left_prompt != right_prompt:
        mismatches.append("prompt")
    left_runtime, right_runtime = left_manifest.get("runtime_kind"), right_manifest.get("runtime_kind")
    if metric == "MODEL_DELTA" and left_runtime != right_runtime:
        mismatches.append("runtime_kind")
    if metric == "HARNESS_GAIN" and {left_runtime, right_runtime} != {"bfcl_protocol", "qwen_harness"} and not (
        left_runtime == "llm_direct" and right_runtime == "qwen_harness"
    ):
        mismatches.append("runtime_pair")
    if metric == "FINAL_DELTA" and not (left_runtime == "bfcl_protocol" and right_runtime == "qwen_harness"):
        mismatches.append("runtime_pair")
    if mismatches:
        raise ValueError(f"runs are not a valid paired comparison; mismatched: {', '.join(mismatches)}")
    left_rows, right_rows = _result_rows(before), _result_rows(after)
    result = paired_gain(
        {key: bool(row.get("correct")) for key, row in left_rows.items()},
        {key: bool(row.get("correct")) for key, row in right_rows.items()},
    )
    result["paired_breakdown"] = _paired_breakdown(left_rows, right_rows)
    if metric == "CPR":
        baseline = result.get("before_accuracy")
        result["context_performance_retention"] = result.get("after_accuracy") / baseline if baseline else None
    result.update({"metric": metric, "before_run": before.name, "after_run": after.name})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute a validated paired capability gain")
    parser.add_argument("metric", choices=["TUG", "SUG", "MUG", "VG", "CPR", "SCG",
                                                   "MODEL_DELTA", "HARNESS_GAIN", "FINAL_DELTA"])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.before, args.after, args.metric), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
