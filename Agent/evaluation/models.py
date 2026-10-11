from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentInput:
    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    attachments: list[dict[str, Any]]
    environment: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Reference:
    id: str
    reference: str
    answer_extractor: str
    evaluator_version: str
    normalization: list[str]
    metadata: dict[str, Any]


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


@dataclass(frozen=True)
class Prediction:
    id: str
    status: str
    answer: str | None
    metadata: dict[str, Any]
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleResult:
    id: str
    status: str
    score: float
    correct: bool
    prediction: str | None
    reference: str
    evaluation_scope: str
    split: str
    metadata: dict[str, Any]
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkMetrics:
    schema_version: str
    benchmark: str
    evaluation_scope: str
    split: str
    total: int
    correct: int
    accuracy: float
    status_counts: dict[str, int]
    by_level: dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]
    by_level_category: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
