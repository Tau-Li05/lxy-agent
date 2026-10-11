from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from Agent.data_pipeline.io import read_jsonl, write_jsonl
from Agent.evaluation.adapters import BoxedLatexOutputAdapter
from Agent.evaluation.aggregation import CompetitionMathAggregator
from Agent.evaluation.checkpoints import load_checkpoints
from Agent.evaluation.math_verify_evaluator import MathVerifyEvaluator
from Agent.evaluation.models import AgentInput, AgentOutput, Prediction, Reference, SampleResult
from Agent.evaluation.runner import execute_run
from Agent.loaders.competition_math import load_competition_math_dev
from Agent.reporting.dev_report import build_dev_report
from Agent.runtime.langchain_runtime import LangChainRuntime


ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = ROOT / "Agent"
DEV_ROOT = ROOT / "data" / "current" / "dev" / "competition_math"


def _input(sample_id: str = "x") -> AgentInput:
    return AgentInput(sample_id, [{"role": "user", "content": "1+1"}], [], [], {}, {"benchmark": "competition_math"})


def _output(sample_id: str = "x", text: str | None = r"work \boxed{2}", status: str = "success") -> AgentOutput:
    return AgentOutput(sample_id, status, text, metadata={"usage_available": False})


def _reference(sample_id: str = "x", answer: str = "2") -> Reference:
    return Reference(sample_id, answer, "boxed_latex", "1", ["strip"], {"benchmark": "competition_math", "split": "validation", "level": "Level 1", "category": "Algebra"})


class FakeGraph:
    def __init__(self, result: Any = None, error: Exception | None = None, delay: float = 0) -> None:
        self.result = result
        self.error = error
        self.delay = delay

    async def ainvoke(self, value: Any, config: Any = None) -> Any:
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def _runtime_config(timeout: float = 1) -> dict[str, Any]:
    return {"runtime": {"limits": {"concurrency": 1, "max_turns": 1, "max_tool_calls": 0, "max_steps": 8, "sample_timeout_seconds": timeout}}}


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_missing_usage(self) -> None:
        runtime = LangChainRuntime(_runtime_config(), FakeGraph({"messages": [{"role": "assistant", "content": r"\boxed{2}"}]}))
        output = await runtime.run(_input())
        self.assertEqual(output.status, "success")
        self.assertFalse(output.metadata["usage_available"])
        self.assertEqual(output.usage["total_tokens"], 0)

    async def test_tool_call_is_limit_exceeded(self) -> None:
        message = {"role": "assistant", "content": "", "tool_calls": [{"name": "x", "args": {}}]}
        output = await LangChainRuntime(_runtime_config(), FakeGraph({"messages": [message]})).run(_input())
        self.assertEqual(output.status, "limit_exceeded")

    async def test_timeout_and_runtime_error(self) -> None:
        timed = await LangChainRuntime(_runtime_config(0.001), FakeGraph({}, delay=0.02)).run(_input())
        failed = await LangChainRuntime(_runtime_config(), FakeGraph(error=RuntimeError("bad"))).run(_input())
        self.assertEqual(timed.status, "timeout")
        self.assertEqual(failed.status, "runtime_error")


class AdapterEvaluatorTests(unittest.TestCase):
    def test_boxed_extraction_and_missing(self) -> None:
        adapter = BoxedLatexOutputAdapter()
        self.assertEqual(adapter.adapt(_output(text=r"a \boxed{\frac{1}{2}} b")).answer, r"\frac{1}{2}")
        self.assertEqual(adapter.adapt(_output(text=r"\boxed 9.")).answer, "9")
        self.assertEqual(adapter.adapt(_output(text="none")).status, "invalid_prediction")

    def test_verify_receives_gold_then_prediction(self) -> None:
        calls: list[tuple[Any, Any]] = []

        def parse(value: str) -> list[str]:
            return [value]

        def verify(gold: Any, prediction: Any) -> bool:
            calls.append((gold, prediction))
            return True

        result = MathVerifyEvaluator(parse, verify).evaluate(Prediction("x", "success", "0.5", {}), _reference(answer=r"\frac{1}{2}"))
        self.assertTrue(result.correct)
        self.assertIn(r"\frac{1}{2}", calls[0][0][0])
        self.assertIn("0.5", calls[0][1][0])

    def test_parse_failure_statuses(self) -> None:
        evaluator = MathVerifyEvaluator(lambda value: [], lambda gold, pred: True)
        self.assertEqual(evaluator.evaluate(Prediction("x", "success", "2", {}), _reference()).status, "evaluator_error")
        calls = iter([["gold"], []])
        evaluator = MathVerifyEvaluator(lambda value: next(calls), lambda gold, pred: True)
        self.assertEqual(evaluator.evaluate(Prediction("x", "success", "bad", {}), _reference()).status, "invalid_prediction")


class LoaderAggregationTests(unittest.TestCase):
    def test_real_dev_loader_and_reference_isolation(self) -> None:
        dataset = load_competition_math_dev(DEV_ROOT / "inputs.jsonl", DEV_ROOT / "references.jsonl", AGENT_ROOT)
        self.assertEqual(len(dataset.inputs), 625)
        self.assertEqual(len(dataset.references), 625)
        self.assertNotIn("reference", dataset.inputs[0].to_dict())
        self.assertEqual(dataset.inputs[0].metadata["split"], "validation")

    def test_id_mismatch_and_wrong_split_fail(self) -> None:
        input_row = next(read_jsonl(DEV_ROOT / "inputs.jsonl"))[1]
        reference_row = next(read_jsonl(DEV_ROOT / "references.jsonl"))[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "inputs.jsonl", [input_row])
            bad = dict(reference_row)
            bad["id"] = "different"
            write_jsonl(root / "references.jsonl", [bad])
            with self.assertRaises(ValueError):
                load_competition_math_dev(root / "inputs.jsonl", root / "references.jsonl", AGENT_ROOT)

    def test_all_failures_remain_in_denominator(self) -> None:
        statuses = ["correct", "incorrect", "timeout", "runtime_error", "limit_exceeded", "invalid_prediction", "evaluator_error"]
        results = [
            SampleResult(str(i), status, float(i == 0), i == 0, None, "2", "development", "validation", {"level": "L", "category": "C"})
            for i, status in enumerate(statuses)
        ]
        metrics = CompetitionMathAggregator().aggregate(results)
        self.assertEqual(metrics.total, 7)
        self.assertEqual(metrics.accuracy, 1 / 7)
        self.assertEqual(sum(group["total"] for group in metrics.by_level.values()), 7)
        report = build_dev_report(metrics)
        self.assertIn("dev_metrics", report)
        self.assertNotIn("test_metrics", report)


class CheckpointTests(unittest.TestCase):
    def test_truncated_final_line_is_ignored_but_middle_corruption_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(json.dumps(_output().to_dict()) + "\n{" , encoding="utf-8")
            self.assertEqual(set(load_checkpoints(path)), {"x"})
            path.write_text("{bad}\n" + json.dumps(_output().to_dict()) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoints(path)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        self.calls += 1
        return _output(agent_input.id)


class FakeEvaluator:
    def evaluate(self, prediction: Prediction, reference: Reference) -> SampleResult:
        return SampleResult(prediction.id, "correct", 1.0, True, prediction.answer, reference.reference, "development", "validation", reference.metadata)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_and_rescore_does_not_call_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            config = temp / "run.yaml"
            config.write_text(
                "\n".join([
                    'schema_version: "1.0"',
                    "run_id: smoke", "benchmark: competition_math", "evaluation_scope: development",
                    f"agent_config: {ROOT / 'Agent/configs/agents/langchain_openai_compatible.yaml'}",
                    "dataset:", f"  inputs: {DEV_ROOT / 'inputs.jsonl'}", f"  references: {DEV_ROOT / 'references.jsonl'}",
                    f"output_root: {temp / 'runs'}", "",
                ]), encoding="utf-8")
            runtime = FakeRuntime()
            run_dir = await execute_run(config, ROOT, limit=5, runtime=runtime, evaluator=FakeEvaluator())
            self.assertEqual(runtime.calls, 5)
            ids = [row["id"] for _, row in read_jsonl(run_dir / "agent_outputs.jsonl")]
            self.assertEqual(ids, sorted(ids))
            forbidden_runtime = FakeRuntime()
            await execute_run(config, ROOT, limit=5, rescore=True, runtime=forbidden_runtime, evaluator=FakeEvaluator())
            self.assertEqual(forbidden_runtime.calls, 0)
            with self.assertRaises(FileExistsError):
                await execute_run(config, ROOT, limit=5, runtime=runtime, evaluator=FakeEvaluator())


if __name__ == "__main__":
    unittest.main()
