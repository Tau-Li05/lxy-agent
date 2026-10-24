from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable


def pass_at_k(rows: Iterable[dict[str, Any]], ks: tuple[int, ...] = (1, 4, 8)) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(bool(row["correct"]))
    result: dict[str, Any] = {}
    for k in ks:
        eligible = [values for values in grouped.values() if len(values) >= k]
        result[f"pass@{k}"] = (
            sum(any(values[:k]) for values in eligible) / len(eligible) if eligible else None
        )
        result[f"pass@{k}_samples"] = len(eligible)
    all_rows = [value for values in grouped.values() for value in values]
    result["mean_sample_accuracy"] = sum(all_rows) / len(all_rows) if all_rows else None
    return result


def reasoning_efficiency(results: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    correct_ids = {row["id"] for row in results if row.get("correct")}
    totals = {"visible_step_tokens": 0, "completion_tokens": 0, "normalized_completion_tokens": 0}
    inference_seconds = 0.0
    for output in outputs:
        usage = output.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        inference_seconds += float((output.get("timing") or {}).get("duration_seconds") or 0)
    correct = len(correct_ids)
    return {
        "correct": correct,
        "re_visible": correct / totals["visible_step_tokens"] if totals["visible_step_tokens"] else None,
        "re_output": correct / totals["completion_tokens"] if totals["completion_tokens"] else None,
        "re_normalized": correct / totals["normalized_completion_tokens"] if totals["normalized_completion_tokens"] else None,
        "re_time": correct / inference_seconds if inference_seconds else None,
        "denominators": {**totals, "inference_seconds": inference_seconds},
    }


def tool_metrics(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results = [row for row in events if row.get("event_type") in {"tool_result", "search_result"}]
    successes = 0
    for row in results:
        result = (row.get("payload") or {}).get("result", {})
        successes += int(bool(result.get("success")) if isinstance(result, dict) else False)
    return {
        "tool_executions": len(results),
        "successful_executions": successes,
        "tool_execution_success_rate": successes / len(results) if results else None,
        "tool_selection_accuracy": None,
        "argument_accuracy": None,
        "unavailable_reason": "selection and semantic argument accuracy require benchmark-specific call annotations",
    }


def paired_gain(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    shared = sorted(left.keys() & right.keys())
    if not shared:
        return {"gain": None, "paired_samples": 0, "unavailable_reason": "no matched samples"}
    before = sum(bool(left[key]) for key in shared) / len(shared)
    after = sum(bool(right[key]) for key in shared) / len(shared)
    differences = [int(bool(right[key])) - int(bool(left[key])) for key in shared]
    rng = random.Random(42)
    boot = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(2000))
    return {"gain": after - before, "before_accuracy": before, "after_accuracy": after,
            "paired_samples": len(shared), "confidence_interval_95": [boot[49], boot[1949]]}


def unavailable_metric(reason: str, *, coverage: float = 0.0) -> dict[str, Any]:
    return {"value": None, "coverage": coverage, "unavailable_reason": reason}


def state_consistency(checkpoints_path: Path) -> dict[str, Any]:
    if not checkpoints_path.exists():
        return {"consistent_checkpoints": 0, "checked_checkpoints": 0, "state_consistency_rate": None}
    from Agent.telemetry.recorder import content_hash

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in checkpoints_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["trajectory_id"])].append(row)
    checked = consistent = 0
    errors: list[dict[str, Any]] = []
    for trajectory, rows in grouped.items():
        previous = None
        for row in rows:
            checked += 1
            reasons = []
            if row.get("previous_hash") != previous:
                reasons.append("previous_hash")
            if row.get("state") is None and row.get("state_artifact"):
                import gzip
                with gzip.open(checkpoints_path.parent / row["state_artifact"]["path"], "rb") as stream:
                    state = json.loads(stream.read().decode("utf-8"))
            else:
                state = row.get("state") or {}
            actual = content_hash(state)
            if row.get("state_hash") != actual:
                reasons.append("state_hash")
            for key in ("model_turn_count", "tool_call_count", "timeout_retry_count",
                        "transport_retry_count", "malformed_retry_count", "output_recovery_count"):
                if key in state and int(state[key]) < 0:
                    reasons.append(f"negative:{key}")
            if reasons:
                errors.append({"trajectory_id": trajectory, "sequence_number": row.get("sequence_number"),
                               "reasons": reasons})
            else:
                consistent += 1
            previous = actual
    return {"consistent_checkpoints": consistent, "checked_checkpoints": checked,
            "state_consistency_rate": consistent / checked if checked else None, "errors": errors}


def event_metrics(events: list[dict[str, Any]], outcomes: dict[str, bool] | None = None,
                  *, planning_enabled: bool = False) -> dict[str, Any]:
    trajectories = {row.get("trajectory_id") for row in events}
    planning = {trajectory: [row for row in events if row.get("trajectory_id") == trajectory
                             and row.get("event_type") in {"plan_created", "plan_revised"}]
                for trajectory in trajectories}
    planned = [trajectory for trajectory, rows in planning.items() if rows]
    revised = [trajectory for trajectory in planned if any(row["event_type"] == "plan_revised" for row in planning[trajectory])]
    revisions = len(revised)
    outcomes = outcomes or {}
    denominator = len(trajectories) if planning_enabled else 0
    successful = sum(bool(outcomes.get(str(trajectory))) for trajectory in trajectories) if planning_enabled else 0
    failed_then_replanned = []
    for trajectory in revised:
        rows = [row for row in events if row.get("trajectory_id") == trajectory]
        if any(row.get("event_type") in {"tool_result", "search_result"}
               and not bool((row.get("payload") or {}).get("result", {}).get("success", True)) for row in rows):
            failed_then_replanned.append(trajectory)
    compactions = [row for row in events if row.get("event_type") == "context_compacted"]
    ratios = [float(row["payload"]["compression_ratio"]) for row in compactions]
    context_actions: dict[str, int] = {}
    for row in compactions:
        for action in (row.get("payload") or {}).get("actions", []):
            level = str(action.get("level", "unknown"))
            context_actions[level] = context_actions.get(level, 0) + 1
    plan_finishes = [row for row in events if row.get("event_type") == "plan_finished"]
    plan_tasks = [task for row in plan_finishes for task in (row.get("payload") or {}).get("tasks", [])]
    completed_tasks = [task for task in plan_tasks if task.get("status") in {"completed", "skipped"}]
    rejected_updates = [row for row in events if row.get("event_type") == "plan_task_updated"
                        and not (row.get("payload") or {}).get("accepted")]
    unplanned_calls = [row for row in events if row.get("event_type") == "unplanned_tool_call"]
    route_events = [row for row in events if row.get("event_type") == "router_decision"]
    route_by_trajectory = {str(row.get("trajectory_id")): (row.get("payload") or {}).get("route")
                           for row in route_events}
    direct_routes = [trajectory for trajectory, route in route_by_trajectory.items() if route == "direct"]
    agent_routes = [trajectory for trajectory, route in route_by_trajectory.items() if route == "agent"]
    return {
        "tools": tool_metrics(events),
        "planning": {"planning_tasks": denominator, "planned_trajectories": len(planned),
                     "replanned_trajectories": revisions,
                     "plan_success_rate": successful / denominator if denominator else None,
                     "replanning_tasks": len(failed_then_replanned),
                     "replanning_success_rate": (
                         sum(bool(outcomes.get(str(value))) for value in failed_then_replanned) / len(failed_then_replanned)
                         if failed_then_replanned else None
                     ),
                     "completed_plans": sum(bool((row.get("payload") or {}).get("completed")) for row in plan_finishes),
                     "plan_completion_rate": (
                         sum(bool((row.get("payload") or {}).get("completed")) for row in plan_finishes) / len(plan_finishes)
                         if plan_finishes else None
                     ),
                     "completed_tasks": len(completed_tasks), "total_tasks": len(plan_tasks),
                     "task_completion_rate": len(completed_tasks) / len(plan_tasks) if plan_tasks else None,
                     "rejected_task_updates": len(rejected_updates),
                     "unplanned_tool_calls": len(unplanned_calls),
                     "planning_efficiency": None,
                     "unavailable_reason": "verified shortest-path annotations are not present"},
        "context": {"compaction_count": len(compactions),
                    "mean_compression_ratio": sum(ratios) / len(ratios) if ratios else None,
                    "actions_by_level": context_actions},
        "routing": {
            "routed_trajectories": len(route_by_trajectory),
            "direct_routed": len(direct_routes), "agent_routed": len(agent_routes),
            "agent_route_rate": len(agent_routes) / len(route_by_trajectory) if route_by_trajectory else None,
            "direct_path_accuracy": (
                sum(bool(outcomes.get(value)) for value in direct_routes) / len(direct_routes) if direct_routes else None
            ),
            "agent_path_accuracy": (
                sum(bool(outcomes.get(value)) for value in agent_routes) / len(agent_routes) if agent_routes else None
            ),
        },
    }
