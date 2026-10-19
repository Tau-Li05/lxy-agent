from __future__ import annotations

import inspect
import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from Agent.evaluation.config import load_run_config, resolve_path
from Agent.evaluation.direct_dataset import load_evaluation_dataset
from Agent.evaluation.io import stable_json
from Agent.runtime.direct_runtime import (
    DirectLangChainRuntime, _execute_locomo_memory, _initialize_memory_db,
    _is_retryable_transport_error,
)
from Agent.evaluation.direct_runner import _locomo_f1, _locomo_multi_answer_f1
from Agent.cli.retry_transient import is_transient_output
from Agent.evaluation.models import AgentOutput

ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = ROOT / "Agent"


class FakeMessage:
    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None, done_reason: str = "stop") -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.done_reason = done_reason

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls,
                "response_metadata": {"done_reason": self.done_reason}}


class FakeModel:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message
        self.calls = 0
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeModel":
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> FakeMessage:
        self.calls += 1
        return self.message


class ResponseError(Exception):
    pass


class RetryModel(FakeModel):
    def __init__(self, failures: int, message: FakeMessage, error_message: str = "XML syntax error on line 3") -> None:
        super().__init__(message)
        self.failures = failures
        self.error_message = error_message

    async def ainvoke(self, messages: list[Any]) -> FakeMessage:
        self.calls += 1
        if self.calls <= self.failures:
            raise ResponseError(self.error_message)
        return self.message


class OpenAIConnectionError(Exception):
    pass


class TransportRetryModel(FakeModel):
    def __init__(self, failures: int, message: FakeMessage, exception_type: type[Exception] = OpenAIConnectionError) -> None:
        super().__init__(message)
        self.failures = failures
        self.exception_type = exception_type

    async def ainvoke(self, messages: list[Any]) -> FakeMessage:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exception_type("Connection error.")
        return self.message


class TimeoutModel(FakeModel):
    async def ainvoke(self, messages: list[Any]) -> FakeMessage:
        self.calls += 1
        await asyncio.sleep(1)
        return self.message


class ConfigDatasetTests(unittest.TestCase):
    def test_locomo_fts5_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            _initialize_memory_db(path)
            stored = _execute_locomo_memory(path, {
                "name": "memory_add", "args": {"content": "Caroline attended an LGBTQ support group",
                                                   "date": "8 May 2023", "source_ids": ["D1:3"]},
            }, read_only=False)
            self.assertIn('"status": "stored"', stored)
            result = _execute_locomo_memory(path, {
                "name": "memory_search", "args": {"query": "Caroline support group", "limit": 5},
            }, read_only=True)
            self.assertIn("LGBTQ support group", result)

    def test_locomo_scoring_matches_expected_behaviour(self) -> None:
        self.assertEqual(_locomo_f1("The support group", "support group"), 1.0)
        self.assertEqual(_locomo_multi_answer_f1("Seattle, hiking", "hiking, Seattle"), 1.0)

    def test_retry_script_selects_only_transient_outputs(self) -> None:
        def output(status: str, kind: str | None = None, message: str = "") -> AgentOutput:
            return AgentOutput("sample", status, None, error={"type": kind, "message": message} if kind else None)

        self.assertTrue(is_transient_output(output("timeout", "timeout", "sample timeout exceeded")))
        self.assertTrue(is_transient_output(output("runtime_error", "OpenAIConnectionError", "Connection error.")))
        self.assertTrue(is_transient_output(output("runtime_error", "OpenAIAPIError", "502 Bad Gateway")))
        self.assertTrue(is_transient_output(output("runtime_error", "OpenAIAPIError", "Internal Server Error")))
        self.assertFalse(is_transient_output(output("limit_exceeded", "output_token_limit", "model output reached max_tokens")))
        self.assertFalse(is_transient_output(output("runtime_error", "AuthenticationError", "invalid API key")))

    def test_only_transient_server_errors_are_retryable(self) -> None:
        class OpenAIAPIError(Exception):
            pass

        self.assertTrue(_is_retryable_transport_error(OpenAIAPIError("502 Bad Gateway")))
        self.assertTrue(_is_retryable_transport_error(OpenAIAPIError("Internal Server Error")))
        self.assertFalse(_is_retryable_transport_error(OpenAIAPIError("invalid API key")))

    def test_stable_json_supports_mixed_mapping_key_types(self) -> None:
        self.assertEqual(stable_json({"outer": {2: "two", "10": "ten"}}), '{"outer":{"10":"ten","2":"two"}}')

    def test_stable_json_represents_backend_objects(self) -> None:
        class BackendState:
            def __repr__(self) -> str:
                return "<backend-state>"

        self.assertEqual(stable_json({"state": BackendState()}), '{"state":"<backend-state>"}')

    def test_all_v2_configs_load_direct_datasets(self) -> None:
        expected = {"gsm8k": 1319, "aime": 90, "bfcl_single_turn": 547,
                    "bfcl_multi_turn": 120, "locomo": 10}
        for name, count in expected.items():
            path = AGENT_ROOT / f"configs/evaluations/{name}.yaml"
            config = load_run_config(path, AGENT_ROOT)
            dataset_path = resolve_path(config["dataset"]["path"], path, ROOT)
            records = load_evaluation_dataset(dataset_path, AGENT_ROOT, config)
            self.assertEqual(len(records), count)
            self.assertNotIn("reference", records[0].input)

    def test_agent_runtime_does_not_import_dataset_processing(self) -> None:
        import Agent.runtime.direct_runtime as runtime
        self.assertNotIn("dataset_processing", inspect.getsource(runtime))

    def test_ollama_receives_explicit_reasoning_false(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        captured: dict[str, Any] = {}

        class FakeChatOllama:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        module = types.ModuleType("langchain_ollama")
        module.ChatOllama = FakeChatOllama
        with patch.dict(sys.modules, {"langchain_ollama": module}):
            DirectLangChainRuntime(config, ROOT, profile_name="qwen3_5_9b")
        self.assertIs(captured["reasoning"], False)
        self.assertEqual(captured["num_ctx"], 65536)

    def test_openai_compatible_disables_thinking_explicitly(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        captured: dict[str, Any] = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        module = types.ModuleType("langchain_openai")
        module.ChatOpenAI = FakeChatOpenAI
        env = {"FIRE_AGENT_BASE_URL": "https://example.invalid/v1", "FIRE_AGENT_API_KEY": "test-key"}
        with patch.dict(sys.modules, {"langchain_openai": module}), patch.dict("os.environ", env, clear=False):
            DirectLangChainRuntime(config, ROOT, profile_name="v4flash")

        self.assertEqual(captured["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("reasoning_effort", captured)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_error_retries_twice_and_records_history(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = TransportRetryModel(2, FakeMessage("#### 18"))

        with patch("Agent.runtime.direct_runtime._sleep_before_retry", return_value=None):
            output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)

        self.assertEqual(output.status, "success")
        self.assertEqual(model.calls, 3)
        self.assertEqual(output.metadata["attempt_count"], 3)
        self.assertEqual(len(output.metadata["retry_events"]), 2)

    async def test_timeout_retries_twice_then_stops(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        config["agent"]["sample_timeout_seconds"] = 0.001
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = TimeoutModel(FakeMessage("unused"))

        with patch("Agent.runtime.direct_runtime._sleep_before_retry", return_value=None):
            output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)

        self.assertEqual(output.status, "timeout")
        self.assertEqual(model.calls, 3)
        self.assertEqual(output.metadata["attempt_count"], 3)
        self.assertEqual(len(output.metadata["retry_events"]), 3)

    async def test_xml_response_error_retries_at_most_twice(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = RetryModel(2, FakeMessage("#### 18"))

        output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)

        self.assertEqual(output.status, "success")
        self.assertEqual(model.calls, 3)
        self.assertEqual(output.metadata["xml_retry_count"], 2)

    async def test_non_xml_response_error_is_not_retried(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = RetryModel(1, FakeMessage("unused"), error_message="connection failed")

        output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)

        self.assertEqual(output.status, "runtime_error")
        self.assertEqual(model.calls, 1)

    async def test_xml_response_error_stops_after_two_retries(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = RetryModel(3, FakeMessage("unused"))

        output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)

        self.assertEqual(output.status, "runtime_error")
        self.assertEqual(model.calls, 3)

    async def test_gsm8k_is_exactly_one_call_without_tools(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = FakeModel(FakeMessage("reasoning\n#### 18"))
        output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.bound_tools, [])
        self.assertEqual(output.final_answer, "reasoning\n#### 18")

    async def test_bfcl_single_turn_binds_schema_but_does_not_execute(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/bfcl_single_turn.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        model = FakeModel(FakeMessage("", [{"id": "c1", "name": record.input["tools"][0]["name"], "args": {}}]))
        output = await DirectLangChainRuntime(config, ROOT, model=model).run(record.input)
        self.assertEqual(model.calls, 1)
        self.assertTrue(model.bound_tools)
        self.assertEqual(len(output.tool_calls), 1)

    async def test_output_length_is_limit_exceeded(self) -> None:
        path = AGENT_ROOT / "configs/evaluations/gsm8k.yaml"
        config = load_run_config(path, AGENT_ROOT)
        record = load_evaluation_dataset(resolve_path(config["dataset"]["path"], path, ROOT), AGENT_ROOT, config)[0]
        output = await DirectLangChainRuntime(config, ROOT, model=FakeModel(FakeMessage("partial", done_reason="length"))).run(record.input)
        self.assertEqual(output.status, "limit_exceeded")
        self.assertEqual(output.final_answer, "partial")
        self.assertEqual(output.metadata["done_reason"], "length")


if __name__ == "__main__":
    unittest.main()
