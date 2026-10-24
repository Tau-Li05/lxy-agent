from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from Agent.telemetry.recorder import content_hash


class AgentPhase(StrEnum):
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class TransitionKind(StrEnum):
    CALL_MODEL = "call_model"
    EXECUTE_TOOLS = "execute_tools"
    VERIFY = "verify"
    COMPACT = "compact"
    RECOVER = "recover"
    FINISH = "finish"
    FAIL = "fail"


@dataclass(frozen=True)
class Transition:
    kind: TransitionKind
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    messages: list[Any]
    phase: AgentPhase = AgentPhase.READY
    transition: Transition | None = None
    model_turn_count: int = 0
    tool_call_count: int = 0
    timeout_retry_count: int = 0
    transport_retry_count: int = 0
    malformed_retry_count: int = 0
    output_recovery_count: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    current_plan: dict[str, Any] | None = None
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    final_answer: str | None = None
    usage: dict[str, int | None] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "provider_reasoning_tokens": None, "visible_step_tokens": 0,
        "normalized_completion_tokens": 0,
    })
    error: dict[str, Any] | None = None

    def digest(self) -> str:
        return content_hash(asdict(self))

