from __future__ import annotations

from typing import Protocol, Sequence

from .models import AgentInput, AgentOutput, BenchmarkMetrics, Prediction, Reference, SampleResult


class AgentRuntime(Protocol):
    async def run(self, agent_input: AgentInput) -> AgentOutput: ...


class OutputAdapter(Protocol):
    def adapt(self, output: AgentOutput) -> Prediction: ...


class Evaluator(Protocol):
    def evaluate(self, prediction: Prediction, reference: Reference) -> SampleResult: ...


class MetricAggregator(Protocol):
    def aggregate(self, results: Sequence[SampleResult]) -> BenchmarkMetrics: ...
