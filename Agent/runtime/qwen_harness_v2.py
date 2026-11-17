"""Isolated v2 experiment: budget warnings plus compact per-sample working memory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Agent.runtime.qwen_harness_v1 import QwenHarnessRuntimeV1


class QwenHarnessRuntimeV2(QwenHarnessRuntimeV1):
    runtime_kind = "qwen_harness_v2"

    async def _run(self, item: dict[str, Any], started: float, attempt: int = 1):
        return await self._run_v2(item, started, attempt)

    async def _run_v2(self, item: dict[str, Any], started: float, attempt: int = 1):
        return await self._run_v1(item, started, attempt)

    def _after_tool_v2(self, memory: dict[str, Any], question: list[Any],
                       evidence: Any, call: dict[str, Any], result: Any) -> None:
        snapshot = evidence.snapshot()
        candidates = [row for row in snapshot["candidates"]
                      if row.get("status") != "failed_partial"]
        values = {row["value"] for row in candidates}
        latest = snapshot["recent_results"][-1] if snapshot["recent_results"] else None
        unresolved = []
        if not candidates:
            unresolved.append("No explicit candidate answer has been established.")
        if len(values) > 1:
            unresolved.append("Candidate answers conflict; reconcile the original constraints.")
        if latest and not latest["success"]:
            unresolved.append("Latest tool call failed; inspect and repair its error.")
        if candidates and not evidence.can_extract:
            unresolved.append("Candidate is not fully checked against the original constraints and search bounds.")
        memory.clear()
        memory.update({
            "problem_constraints": [str(row.get("content", "")) for row in question if row.get("role") == "user"],
            "candidate_answer": candidates[-1]["value"] if len(values) == 1 else None,
            "candidate_status": candidates[-1]["status"] if len(values) == 1 else None,
            "candidate_eligible": evidence.can_extract,
            "candidate_reviewable": evidence.can_review_candidate,
            "tool_evidence": deepcopy(snapshot["recent_results"][-4:]),
            "unverified_points": unresolved,
        })

    _after_tool_v1 = _after_tool_v2

    def _final_messages_v2(self, evidence: Any, question: list[Any],
                           partial: str | None, history: list[Any], allowance: int,
                           strategy: str, memory: dict[str, Any]) -> list[Any]:
        if not memory:
            return super()._final_messages_v1(evidence, question, partial, history,
                                                allowance, strategy, memory)
        messages = evidence.final_messages(
            self.config["prompt"]["system"], question, partial,
            history=[], max_input_tokens=allowance, strategy=strategy,
            protected_state={"working_memory_v2": memory},
        )
        evidence.finalization_info["working_memory_v2"] = True
        return messages

    _final_messages_v1 = _final_messages_v2
