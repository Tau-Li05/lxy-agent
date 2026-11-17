"""Offline prompt reconstruction; never invokes a model or reads scoring labels."""
from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.engine import estimate_tokens
from Agent.runtime.math_workflow import MathEvidence


def _expand(run_dir: Path, value: Any) -> Any:
    if isinstance(value, dict) and "artifact" in value:
        path = (run_dir / value["artifact"]["path"]).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("artifact path outside run")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return value


def replay(run_dir: Path) -> dict[str, Any]:
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    config = resolved["config"]
    outputs = [json.loads(line) for line in (run_dir / "agent_outputs.jsonl").open()]
    final_ids = {row["metadata"].get("trajectory_id"): row["id"] for row in outputs}
    samples = {row["id"]: row["input"] for row in map(json.loads, (run_dir / "samples.jsonl").open())}
    grouped: dict[str, list[Any]] = {}
    for event in map(json.loads, (run_dir / "events.jsonl").open()):
        trajectory = event["trajectory_id"]
        if trajectory in final_ids:
            grouped.setdefault(trajectory, []).append(event)
    rows = []
    profile = config["model"]["profiles"][resolved["profile"]]
    output_cap = int(config["features"]["finalization"].get("max_tokens", 2048))
    for trajectory, events in grouped.items():
        starts = [i for i, event in enumerate(events) if event["event_type"] == "finalization_started"]
        if not starts:
            continue
        index = starts[0]
        if any(e["event_type"] == "context_compacted" for e in events[:index]):
            raise ValueError("replay requires an uncompacted original trajectory")
        question = samples[final_ids[trajectory]]["turns"][0]
        history = deepcopy(question)
        ledger = MathEvidence()
        latest = None
        # Rebuild from typed events, not repr strings in old request checkpoints.
        # Planning-tool replies are deliberately omitted; no mathematics is fabricated.
        for event in events[:index]:
            payload = event["payload"]
            if event["event_type"] == "model_response":
                response = payload["response"]
                history.append({"role": "assistant", "content": response.get("content", ""),
                                "tool_calls": response.get("tool_calls", [])})
                latest = response.get("content")
            elif event["event_type"] == "tool_result":
                result = payload["result"]
                if "artifact" in result:
                    result = _expand(run_dir, result)
                history.append({"role": "tool", "tool_call_id": payload["call"]["id"],
                                "content": result.get("output", "")})
        evidence_event = next((e for e in events[index:]
                               if e["event_type"] == "finalization_evidence"), None)
        if evidence_event:
            snapshot = evidence_event["payload"]
            ledger.candidates = deepcopy(snapshot.get("candidates", []))
            ledger.results = deepcopy(snapshot.get("recent_results", []))
        remaining = events[index]["payload"].get("remaining_tokens")
        if not isinstance(remaining, int):
            raise ValueError("remaining budget is unavailable or redacted")
        allowance = max(0, min(remaining, profile["context_window"]) - output_cap - 128)
        messages = ledger.final_messages(config["prompt"]["system"], question, latest,
                                         history=history, max_input_tokens=allowance)
        old_request = next(e for e in events[index:] if e["event_type"] == "model_request")
        old_messages = _expand(run_dir, old_request["payload"]["messages"])
        info = dict(ledger.finalization_info)
        info.update(id=final_ids[trajectory], original_question_preserved=all(q in messages for q in question),
                    old_input_tokens_estimate=int(1.25 * estimate_tokens([*old_messages, {"tools": []}])) + 128,
                    fits_input_budget=info["input_tokens_estimate"] <= allowance)
        rows.append(info)
    return {"run": run_dir.name, "mode": "offline_prompt_replay_no_model_no_labels",
            "cases": len(rows), "extraction_cases": sum(r["mode"] == "extract" for r in rows),
            "all_fit_input_budget": all(r["fits_input_budget"] for r in rows),
            "all_original_questions_preserved": all(r["original_question_preserved"] for r in rows),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
