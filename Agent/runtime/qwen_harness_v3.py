"""Isolated v3 experiment: v1 budget notices plus memory and recent reasoning."""

from __future__ import annotations

from typing import Any

from Agent.runtime.qwen_harness_v2 import QwenHarnessRuntimeV2


class QwenHarnessRuntimeV3(QwenHarnessRuntimeV2):
    runtime_kind = "qwen_harness_v3"

    async def _run(self, item: dict[str, Any], started: float, attempt: int = 1):
        return await self._run_v3(item, started, attempt)

    async def _run_v3(self, item: dict[str, Any], started: float, attempt: int = 1):
        return await self._run_v1(item, started, attempt)

    def _final_messages_v3(self, evidence: Any, question: list[Any],
                           partial: str | None, history: list[Any], allowance: int,
                           strategy: str, memory: dict[str, Any]) -> list[Any]:
        if not memory:
            return super()._final_messages_v2(evidence, question, partial, history,
                                                allowance, strategy, memory)
        if evidence.can_extract or evidence.can_review_candidate:
            return super()._final_messages_v2(evidence, question, partial, history,
                                                allowance, strategy, memory)
        # An unverified candidate or no candidate needs reasoning continuity. The
        # base evidence builder keeps whole tool exchanges and trims oldest groups.
        # Limit history input so the final output reserve remains protected.
        history_allowance = min(allowance, 12_000)
        messages = evidence.final_messages(
            self.config["prompt"]["system"], question, partial,
            history=history, max_input_tokens=history_allowance,
            strategy=strategy, protected_state={"working_memory_v2": memory},
        )
        evidence.finalization_info["working_memory_v3"] = "memory_plus_recent_history"
        return messages

    _final_messages_v1 = _final_messages_v3
