from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from Agent.evaluation.capability_metrics import paired_gain, pass_at_k
from Agent.evaluation.config import load_run_config, resolve_path
from Agent.evaluation.direct_dataset import load_evaluation_dataset
from Agent.evaluation.models import AgentOutput
from Agent.runtime.qwen_harness import (
    QwenHarnessRuntime, _build_plan, _plan_complete, _runnable_task_ids, _update_plan_task,
)
from Agent.runtime import BFCLProtocolRuntime, DirectRuntime
from Agent.runtime.bfcl_executor import BFCLProtocolExecutor
from Agent.runtime.direct_runtime import (
    _invoke_with_xml_retry, _is_retryable_transport_error,
    _is_retryable_xml_response_error, _openai_tool,
)
from Agent.telemetry import RunRecorder, latest_checkpoint, load_checkpoint_chain, reconstruct_state, state_delta
from Agent.tools.math_tools import execute_math_tool
from Agent.evaluation.reasoning_steps import evaluate_steps
from Agent.engine import ContextManager, ContextPolicy, estimate_tokens, execute_parallel
from Agent.query_engine import (
    Attachment, CostPolicy, PermissionPolicy, QueryBudgetExceeded, QueryEngine, QueryHooks, QueryOptions,
)
from Agent.cli.compare_runs import compare
from Agent.evaluation.direct_runner import _numeric_equivalent


ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = ROOT / "Agent"


class FakeMessage:
    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None,
                 usage: dict[str, int] | None = None, finish_reason: str = "stop") -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        self.response_metadata = {"finish_reason": finish_reason}

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"content": self.content, "tool_calls": self.tool_calls,
                "usage_metadata": self.usage_metadata, "response_metadata": self.response_metadata}


class SequenceModel:
    def __init__(self, *responses: FakeMessage) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.bind_calls = 0
        self.bound_tools: list[Any] = []
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "SequenceModel":
        self.bind_calls += 1
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> FakeMessage:
        self.invocations.append(deepcopy(messages))
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class RuntimeV3Tests(unittest.IsolatedAsyncioTestCase):
    async def test_ollama_function_parameter_error_gets_corrective_retry(self) -> None:
        class ResponseError(Exception):
            status_code = 500

        class MalformedOnceModel:
            def __init__(self) -> None:
                self.messages: list[list[Any]] = []

            async def ainvoke(self, messages: list[Any]) -> FakeMessage:
                self.messages.append(deepcopy(messages))
                if len(self.messages) == 1:
                    raise ResponseError("expected element type <function> but have <parameter> (status code: 500)")
                return FakeMessage("recovered")

        model = MalformedOnceModel()
        response, retries = await _invoke_with_xml_retry(model, [{"role": "user", "content": "call it"}], 2)
        self.assertEqual(response.content, "recovered")
        self.assertEqual(retries, 1)
        self.assertIn("identify the function", str(model.messages[1]))
        error = ResponseError("expected element type <function> but have <parameter> (status code: 500)")
        self.assertTrue(_is_retryable_xml_response_error(error))
        self.assertFalse(_is_retryable_transport_error(error))

    def test_bfcl_tool_schema_is_openai_compatible(self) -> None:
        converted = _openai_tool({
            "name": "convert", "description": "convert value",
            "parameters": {"type": "dict", "properties": {
                "value": {"type": "float"}, "flags": {"type": "list", "items": {"type": "bool"}},
            }},
            "response": {"type": "dict"},
        })
        function = converted["function"]
        self.assertEqual(set(function), {"name", "description", "parameters"})
        self.assertEqual(function["parameters"]["type"], "object")
        self.assertEqual(function["parameters"]["properties"]["value"]["type"], "number")
        self.assertEqual(function["parameters"]["properties"]["flags"]["type"], "array")
        self.assertEqual(function["parameters"]["properties"]["flags"]["items"]["type"], "boolean")

    def test_harness_accepts_openai_compatible_profile(self) -> None:
        path = AGENT_ROOT / "configs/v3/aime_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        fake_model = SequenceModel(FakeMessage("DIRECT"), FakeMessage("#### 1"))
        with patch.object(QwenHarnessRuntime, "_create_model", return_value=fake_model):
            runtime = QwenHarnessRuntime(config, ROOT, profile_name="v4flash")
        self.assertIs(runtime.model, fake_model)
        self.assertEqual(runtime.profile["provider"], "openai_compatible")

    async def test_query_engine_stream_forwards_runtime_events_and_output(self) -> None:
        class StreamingRuntime:
            async def run(self, item: dict[str, Any]) -> AgentOutput:
                sink = item["environment"]["_query_event_sink"]
                await sink({"event": "tool_started", "tool_call_id": "call-1"})
                await sink({"event": "tool_completed", "tool_call_id": "call-1", "success": True})
                return AgentOutput(id=item["id"], status="success", final_answer="done")

        with tempfile.TemporaryDirectory() as directory:
            engine = QueryEngine(StreamingRuntime(), state_root=Path(directory))
            events = [event async for event in engine.stream(
                {"id": "sample", "environment": {}, "turns": [[]]},
                QueryOptions(session_id="stream", persist_session=False),
            )]
        self.assertEqual([event["event"] for event in events], [
            "query_started", "tool_started", "tool_completed", "query_completed",
        ])
        self.assertEqual(events[-1]["output"].final_answer, "done")

    async def test_parallel_tools_stream_completion_but_preserve_request_order(self) -> None:
        calls = [
            {"id": "slow", "name": "wait", "args": {"delay": 0.03}},
            {"id": "fast", "name": "wait", "args": {"delay": 0.001}},
            {"id": "failed", "name": "fail", "args": {"delay": 0.002}},
        ]
        events: list[dict[str, Any]] = []

        async def worker(call: dict[str, Any]) -> str:
            await asyncio.sleep(call["args"]["delay"])
            if call["name"] == "fail":
                raise RuntimeError("isolated failure")
            return call["id"]

        results = await execute_parallel(calls, worker, max_concurrency=3, event_sink=events.append)
        self.assertEqual([result.call["id"] for result in results], ["slow", "fast", "failed"])
        completed = [event["tool_call_id"] for event in events if event["event"] == "tool_completed"]
        self.assertEqual(completed[-1], "slow")
        self.assertIsInstance(results[2].value, RuntimeError)

    async def test_query_engine_lifecycle_dedup_attachments_budget_and_stop_hook(self) -> None:
        path = AGENT_ROOT / "configs/v3/gsm8k_direct.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(FakeMessage("draft\n#### 1"), FakeMessage("fixed\n#### 576"))
        stop_calls = 0

        def stop_hook(output: Any, _options: QueryOptions) -> bool | str:
            nonlocal stop_calls
            stop_calls += 1
            return "Recheck and provide the requested final answer." if stop_calls == 1 else True

        with tempfile.TemporaryDirectory() as directory:
            runtime = DirectRuntime(config, ROOT, profile_name="qwen3_5_9b", model=model)
            runtime.run_state_root = Path(directory) / "runtime"
            engine = QueryEngine(runtime, state_root=Path(directory), hooks=QueryHooks(stop=stop_hook))
            options = QueryOptions(
                session_id="session", message_id="message", persist_session=True,
                attachments=(Attachment("note.txt", "important context"),),
                cost=CostPolicy(max_session_tokens=1000, input_cost_per_million=1, output_cost_per_million=2),
            )
            output = await engine.submit(record.input, options)
            duplicate = await engine.submit(record.input, options)
            session_files = list((Path(directory) / "sessions").glob("*.jsonl"))
        self.assertEqual(model.calls, 2)
        self.assertIs(output, duplicate)
        self.assertEqual(output.metadata["query_engine"]["stop_hook_retries"], 1)
        self.assertIn("important context", str(model.invocations[0]))
        self.assertIn("Recheck", str(model.invocations[1]))
        self.assertEqual(len(session_files), 1)
        self.assertGreater(engine.session_usage("session").estimated_cost, 0)
        with self.assertRaises(QueryBudgetExceeded):
            await engine.submit(record.input, QueryOptions(
                session_id="session", message_id="next", persist_session=False,
                cost=CostPolicy(max_session_tokens=1),
            ))

    async def test_query_engine_request_permissions_remove_tools(self) -> None:
        path = AGENT_ROOT / "configs/v3/bfcl_single_direct.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(FakeMessage("no tools"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = DirectRuntime(config, ROOT, profile_name="qwen3_5_9b", model=model)
            runtime.run_state_root = Path(directory) / "runtime"
            engine = QueryEngine(runtime, state_root=Path(directory))
            await engine.submit(record.input, QueryOptions(
                session_id="s", persist_session=False,
                permissions=PermissionPolicy(allow_tools=False, allow_tool_execution=False),
            ))
        self.assertEqual(model.bind_calls, 0)

    def test_context_manager_applies_auditable_reduction_levels(self) -> None:
        manager = ContextManager(ContextPolicy(
            enabled=True, trigger_ratio=0.50, target_ratio=0.30,
            tool_result_max_characters=100, tool_result_preview_characters=20,
            preserve_recent_messages=2, collapse_summary_characters=200,
        ), context_window=500, reserved_output_tokens=100)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old question " * 40},
            {"role": "assistant", "content": "old answer " * 40},
            {"role": "tool", "tool_call_id": "x", "content": "z" * 500},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        artifacts = []
        result = manager.compact(messages, artifact_writer=lambda value: artifacts.append(value) or {"path": "a"})
        levels = {action["level"] for action in result.actions}
        self.assertTrue(manager.needs_compaction(messages))
        self.assertIn("tool_result_budget", levels)
        self.assertIn("history_snip", levels)
        self.assertIn("context_collapse", levels)
        self.assertLess(result.compacted_tokens, result.original_tokens)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(messages[3]["content"], "z" * 500)
        self.assertGreater(estimate_tokens(messages), 0)

    async def test_locomo_full_context_uses_context_manager_per_question(self) -> None:
        path = AGENT_ROOT / "configs/v3/locomo_context_managed_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        config["limits"]["context_window_override"] = 9000
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(FakeMessage("short answer"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
            events = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(output.status, "success")
        self.assertGreater(output.metadata["compaction_count"], 0)
        context_events = [row for row in events if row["event_type"] == "context_compacted"]
        self.assertTrue(context_events)
        self.assertTrue(all((row["payload"] or {}).get("question_id") for row in context_events))

    def test_structured_plan_enforces_dependencies_and_completion(self) -> None:
        settings = {"max_tasks": 4, "max_replans": 1, "require_verification_task": True}
        plan, error = _build_plan({
            "goal": "solve",
            "tasks": [
                {"id": "derive", "subject": "Derive", "description": "derive expression",
                 "kind": "analysis", "blocked_by": []},
                {"id": "calculate", "subject": "Calculate", "description": "calculate value",
                 "kind": "execution", "blocked_by": ["derive"]},
                {"id": "verify", "subject": "Verify", "description": "verify answer",
                 "kind": "verification", "blocked_by": ["calculate"]},
            ],
        }, None, settings)
        self.assertIsNone(error)
        self.assertEqual(_runnable_task_ids(plan), ["derive"])
        plan, message = _update_plan_task(plan, {"task_id": "calculate", "status": "in_progress"})
        self.assertIn("blocked", message)
        plan, _ = _update_plan_task(plan, {"task_id": "derive", "status": "in_progress"})
        plan, _ = _update_plan_task(plan, {"task_id": "derive", "status": "completed", "result": "x=42"})
        self.assertEqual(_runnable_task_ids(plan), ["calculate"])
        for task_id in ("calculate", "verify"):
            plan, _ = _update_plan_task(plan, {"task_id": task_id, "status": "in_progress"})
            plan, _ = _update_plan_task(plan, {"task_id": task_id, "status": "completed", "result": "ok"})
        self.assertTrue(_plan_complete(plan))

    def test_structured_plan_rejects_cycles_and_missing_verification(self) -> None:
        settings = {"max_tasks": 4, "max_replans": 1, "require_verification_task": True}
        _, cycle = _build_plan({"goal": "x", "tasks": [
            {"id": "a", "subject": "A", "description": "A", "kind": "analysis", "blocked_by": ["b"]},
            {"id": "b", "subject": "B", "description": "B", "kind": "execution", "blocked_by": ["a"]},
        ]}, None, settings)
        self.assertIn("cycle", cycle or "")
        _, verification = _build_plan({"goal": "x", "tasks": [
            {"id": str(i), "subject": str(i), "description": str(i), "kind": "analysis", "blocked_by": []}
            for i in range(3)
        ]}, None, settings)
        self.assertIn("verification", verification or "")

    def test_gsm8k_numeric_equivalence_uses_exact_decimal_values(self) -> None:
        self.assertTrue(_numeric_equivalent("38", "38.0"))
        self.assertTrue(_numeric_equivalent("12.00", "12"))
        self.assertTrue(_numeric_equivalent("-0", "0"))
        self.assertTrue(_numeric_equivalent("1200", "1200.0"))
        self.assertFalse(_numeric_equivalent("-14", "14"))
        self.assertFalse(_numeric_equivalent(None, "0"))

    def test_only_v3_and_canonical_runtime_names_are_accepted(self) -> None:
        source = yaml.safe_load((AGENT_ROOT / "configs/v3/aime_direct.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            source["schema_version"] = "2.0"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only schema_version 3.0"):
                load_run_config(path, AGENT_ROOT)
            source["schema_version"] = "3.0"
            source["runtime"]["kind"] = "deepseek_direct"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_run_config(path, AGENT_ROOT)

    def test_checkpoint_plus_delta_reconstructs_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = RunRecorder(root, experiment_id="e", runtime_kind="qwen_harness",
                                   model_profile="q", sample_id="s", trajectory_id="t")
            before = {"turn_count": 1, "final_answer": None}
            after = {"turn_count": 2, "final_answer": "42"}
            recorder.checkpoint(before, kind="model_response")
            recorder.event("transition", {"state_delta": state_delta(before, after)},
                           state_before_hash=__import__("Agent.telemetry.recorder", fromlist=["content_hash"]).content_hash(before),
                           state_after_hash=__import__("Agent.telemetry.recorder", fromlist=["content_hash"]).content_hash(after))
            self.assertEqual(reconstruct_state(root / "checkpoints.jsonl", root / "events.jsonl",
                                               trajectory_id="t"), after)

    def test_v3_configs_and_web_dataset(self) -> None:
        for path in sorted((AGENT_ROOT / "configs/v3").glob("*.yaml")):
            config = load_run_config(path, AGENT_ROOT)
            records = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)
            self.assertTrue(records)
        web = ROOT / "data/evaluation/bfcl_web_search.jsonl"
        rows = [json.loads(line) for line in web.read_text().splitlines()]
        self.assertEqual(len(rows), 100)
        self.assertEqual({row["input"]["environment"]["initial_config"]["WebSearchAPI"]["show_snippet"] for row in rows}, {True, False})

    def test_checkpoint_hash_chain_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RunRecorder(root, experiment_id="e", runtime_kind="qwen_harness",
                                   model_profile="4b", sample_id="s", trajectory_id="t")
            recorder.checkpoint({"value": 1}, kind="model_response")
            recorder.checkpoint({"value": 2}, kind="finish")
            self.assertEqual(len(load_checkpoint_chain(root / "checkpoints.jsonl", trajectory_id="t")), 2)
            self.assertEqual(latest_checkpoint(root / "checkpoints.jsonl", trajectory_id="t"), {"value": 2})

    def test_capability_metrics(self) -> None:
        rows = [{"sample_id": "a", "correct": False}, {"sample_id": "a", "correct": True},
                {"sample_id": "b", "correct": False}, {"sample_id": "b", "correct": False}]
        self.assertEqual(pass_at_k(rows, (1, 2))["pass@2"], 0.5)
        self.assertEqual(paired_gain({"a": False}, {"a": True})["gain"], 1.0)

    def test_bfcl_execution_ledger_replays_success(self) -> None:
        calls = 0
        def backend(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
            nonlocal calls
            calls += 1
            return ['{"ok": true}'], {}
        executor = BFCLProtocolExecutor(ROOT, backend=backend)
        item = {"id": "x", "environment": {"initial_config": {}, "involved_classes": []},
                "metadata": {"category": "multi_turn_base"}}
        first = executor.execute_batch(["f()"], item=item, trajectory_id="t", tool_call_ids=["c"], attempt=1)
        second = executor.execute_batch(["f()"], item=item, trajectory_id="t", tool_call_ids=["c"], attempt=2)
        self.assertEqual(calls, 1)
        self.assertFalse(first[0].replayed)
        self.assertTrue(second[0].replayed)

    def test_math_tool_blocks_file_and_network_imports(self) -> None:
        self.assertFalse(execute_math_tool("python_calculate", {"code": "import socket"}).success)
        self.assertFalse(execute_math_tool("python_calculate", {"code": "open('/tmp/x', 'w')"}).success)
        self.assertTrue(execute_math_tool("python_calculate", {"code": "print(6 * 7)"}).success)

    def test_visible_step_evaluator_reports_coverage(self) -> None:
        result = evaluate_steps("Step 1: 6 * 7 = 42\nStep 2: therefore the answer follows")
        self.assertEqual(result["correct_steps"], 1)
        self.assertEqual(result["evaluated_steps"], 1)
        self.assertEqual(result["total_steps"], 2)

    async def test_direct_is_one_call_for_openai_and_ollama_profiles(self) -> None:
        path = AGENT_ROOT / "configs/v3/aime_direct.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        for profile in ("v4flash", "qwen3_5_9b"):
            model = SequenceModel(FakeMessage("steps\n#### 18"))
            with tempfile.TemporaryDirectory() as directory:
                runtime = DirectRuntime(config, ROOT, profile_name=profile, model=model)
                runtime.run_state_root = Path(directory)
                output = await runtime.run(record.input)
            self.assertEqual(model.calls, 1)
            self.assertEqual(output.status, "success")
            self.assertEqual(output.metadata["runtime"], "llm_direct")
            self.assertEqual(output.metadata["provider"], config["model"]["profiles"][profile].get("provider"))

    async def test_bfcl_single_direct_returns_call_without_execution(self) -> None:
        path = AGENT_ROOT / "configs/v3/bfcl_single_direct.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        call = {"id": "call-1", "name": record.input["tools"][0]["name"], "args": {}}
        model = SequenceModel(FakeMessage("", [call]))
        with tempfile.TemporaryDirectory() as directory:
            runtime = DirectRuntime(config, ROOT, profile_name="qwen3_5_9b", model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
        self.assertEqual(model.calls, 1)
        self.assertEqual(output.tool_calls, [call])
        self.assertFalse(output.metadata["tool_execution"])

    def test_protocol_configs_are_minimal_and_provider_neutral(self) -> None:
        path = AGENT_ROOT / "configs/v3/bfcl_multi_protocol.yaml"
        config = load_run_config(path, AGENT_ROOT)
        self.assertEqual(set(config["model"]["profiles"]), {"v4flash", "qwen3_5_4b", "qwen3_5_9b"})
        for name in ("planning", "verification", "compaction", "self_correction", "output_recovery"):
            self.assertFalse(config["features"][name]["enabled"])
        first = BFCLProtocolRuntime(config, ROOT, profile_name="v4flash", model=SequenceModel(FakeMessage("done")))
        second = BFCLProtocolRuntime(config, ROOT, profile_name="qwen3_5_9b", model=SequenceModel(FakeMessage("done")))
        self.assertIs(type(first.bfcl_executor), type(second.bfcl_executor))

    def test_comparison_contract_allows_model_delta_and_harness_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def make_run(name: str, runtime: str, model: str) -> Path:
                run = root / name
                run.mkdir()
                manifest = {"runtime_kind": runtime, "exact_model_name": model,
                            "resource_budget": {"max_model_turns": 20}, "dataset_sha256": "same",
                            "resolved_config": {"prompt": {"system": "same"}}}
                (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (run / "sample_results.jsonl").write_text(
                    json.dumps({"id": "sample", "correct": name != "deepseek"}) + "\n", encoding="utf-8")
                return run
            deepseek = make_run("deepseek", "bfcl_protocol", "DeepSeek-V4-Flash")
            qwen_protocol = make_run("qwen-protocol", "bfcl_protocol", "qwen3.5:9b")
            qwen_harness = make_run("qwen-harness", "qwen_harness", "qwen3.5:9b")
            self.assertEqual(compare(deepseek, qwen_protocol, "MODEL_DELTA")["gain"], 1.0)
            harness_gain = compare(qwen_protocol, qwen_harness, "HARNESS_GAIN")
            self.assertEqual(harness_gain["gain"], 0.0)
            self.assertEqual(harness_gain["paired_breakdown"]["transitions"]["incorrect_to_incorrect"], 0)

    async def test_qwen_profiles_share_math_harness(self) -> None:
        path = AGENT_ROOT / "configs/v3/aime_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(
            FakeMessage("AGENT"),
            FakeMessage("", [{"id": "c1", "name": "python_calculate", "args": {"code": "print(18)"}}]),
            FakeMessage("checked\n#### 18"),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, profile_name="qwen3_5_4b", model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
        self.assertEqual(model.calls, 3)
        self.assertEqual(output.status, "success")
        self.assertEqual(output.metadata["tool_call_count"], 1)
        self.assertEqual(output.metadata["profile"], "qwen3_5_4b")

    async def test_qwen_harness_reserves_last_turn_for_final_answer(self) -> None:
        path = AGENT_ROOT / "configs/v3/gsm8k_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        config["limits"]["max_model_turns"] = 2
        config["features"]["tools"] = {"enabled": True, "execution": True}
        config["features"]["difficulty_router"] = {"enabled": False}
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(
            FakeMessage("", [{"id": "c1", "name": "python_calculate", "args": {"code": "print(42)"}}]),
            FakeMessage("Using the tool result.\n#### 42"),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
        self.assertEqual(output.status, "success")
        self.assertEqual(model.calls, 2)
        self.assertTrue(output.metadata["finalization_started"])
        self.assertEqual(model.bind_calls, 1)

    async def test_qwen_harness_executes_structured_plan_tasks(self) -> None:
        path = AGENT_ROOT / "configs/v3/aime_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        config["features"]["difficulty_router"] = {"enabled": False}
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        call = lambda call_id, name, args: FakeMessage("", [{"id": call_id, "name": name, "args": args}])
        model = SequenceModel(
            call("p", "create_plan", {"goal": "solve", "tasks": [
                {"id": "derive", "subject": "Derive", "description": "derive", "kind": "analysis", "blocked_by": []},
                {"id": "calculate", "subject": "Calculate", "description": "calculate", "kind": "execution",
                 "blocked_by": ["derive"]},
            ]}),
            call("s1", "task_update", {"task_id": "derive", "status": "in_progress"}),
            call("d1", "task_update", {"task_id": "derive", "status": "completed", "result": "expression"}),
            call("s2", "task_update", {"task_id": "calculate", "status": "in_progress"}),
            call("m", "python_calculate", {"code": "print(294)"}),
            call("d2", "task_update", {"task_id": "calculate", "status": "completed", "result": "294"}),
            FakeMessage("#### 294"),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
            events = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(output.status, "success")
        self.assertTrue(output.metadata["plan_completed"])
        self.assertEqual([task["status"] for task in output.metadata["plan"]["tasks"]],
                         ["completed", "completed"])
        self.assertEqual(output.final_answer, "#### 294")
        transitions = [(row["payload"]["from"], row["payload"]["to"])
                       for row in events if row["event_type"] == "transition"]
        self.assertIn(("call_model", "execute_tools"), transitions)
        self.assertIn(("execute_tools", "call_model"), transitions)
        self.assertEqual(transitions[-1][1], "finish")

    async def test_transition_state_machine_verifies_and_recovers(self) -> None:
        path = AGENT_ROOT / "configs/v3/gsm8k_qwen_harness.yaml"
        base = load_run_config(path, AGENT_ROOT)
        base["features"]["difficulty_router"] = {"enabled": False}
        record = load_evaluation_dataset(resolve_path(base["dataset"]["path"], path, ROOT), AGENT_ROOT, base)[0]

        verification_config = deepcopy(base)
        verification_config["features"]["planning"] = {"enabled": False}
        verification_config["features"]["verification"] = {"enabled": True, "method": "self"}
        verification_model = SequenceModel(FakeMessage("#### 42"), FakeMessage("VERIFICATION: CORRECT\n#### 42"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(verification_config, ROOT, model=verification_model)
            runtime.run_state_root = Path(directory)
            verified = await runtime.run(record.input)
            verification_events = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(verified.status, "success")
        self.assertIsNotNone(verified.metadata["verification_result"])
        self.assertTrue(any(row["event_type"] == "transition" and row["payload"]["to"] == "verify"
                            for row in verification_events))

        recovery_config = deepcopy(base)
        recovery_config["features"]["planning"] = {"enabled": False}
        recovery_config["features"]["finalization"] = {"enabled": False}
        recovery_config["features"]["output_recovery"] = {"enabled": True, "max_retries": 1}
        recovery_model = SequenceModel(FakeMessage("partial", finish_reason="length"), FakeMessage("#### 42"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(recovery_config, ROOT, model=recovery_model)
            runtime.run_state_root = Path(directory)
            recovered = await runtime.run(record.input)
            recovery_events = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(recovered.status, "success")
        self.assertEqual(recovered.final_answer, "#### 42")
        self.assertTrue(any(row["event_type"] == "transition" and row["payload"]["to"] == "recover"
                            for row in recovery_events))

    async def test_difficulty_router_uses_judgment_without_confidence(self) -> None:
        path = AGENT_ROOT / "configs/v3/gsm8k_adaptive_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(FakeMessage("DIRECT"), FakeMessage("#### 576"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
            events = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(output.status, "success")
        self.assertEqual(output.metadata["route"], "direct")
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.bind_calls, 0)
        decision = next(row for row in events if row["event_type"] == "router_decision")
        self.assertNotIn("confidence", decision["payload"])
        self.assertEqual(decision["payload"]["route"], "direct")

        agent_model = SequenceModel(FakeMessage("AGENT"), FakeMessage("#### 576"))
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=agent_model)
            runtime.run_state_root = Path(directory)
            agent_output = await runtime.run(record.input)
        self.assertEqual(agent_output.metadata["route"], "agent")
        self.assertEqual(agent_model.bind_calls, 1)
        bound_names = {tool["function"]["name"] for tool in agent_model.bound_tools}
        self.assertTrue({"create_plan", "task_list", "task_update", "python_calculate"} <= bound_names)

    async def test_qwen_multiturn_invokes_each_user_turn(self) -> None:
        path = AGENT_ROOT / "configs/v3/bfcl_multi_qwen_harness.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = SequenceModel(*[FakeMessage("turn complete") for _ in record.input["turns"]])
        with tempfile.TemporaryDirectory() as directory:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(directory)
            output = await runtime.run(record.input)
        self.assertEqual(model.calls, len(record.input["turns"]))
        self.assertEqual(len(output.metadata["bfcl_calls_by_turn"]), len(record.input["turns"]))


if __name__ == "__main__":
    unittest.main()
