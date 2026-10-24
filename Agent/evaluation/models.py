from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentOutput:
    id: str
    status: str
    final_answer: str | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    timing: dict[str, Any] = field(default_factory=lambda: {"duration_seconds": 0.0})
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentOutput":
        return cls(**value)
