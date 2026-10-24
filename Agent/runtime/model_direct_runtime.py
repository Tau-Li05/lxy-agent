from __future__ import annotations

import asyncio
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.evaluation.models import AgentOutput
from Agent.prompting import SystemPromptBuilder
from Agent.runtime.direct_runtime import (
    DirectLangChainRuntime, _content_text, _done_reason, _is_retryable_transport_error,
    _openai_tool, _response_usage, _serialize_message, _sleep_before_retry, _tool_calls,
)
from Agent.telemetry import RunRecorder


class DirectRuntime(DirectLangChainRuntime):
    """Provider-neutral one-generation runtime with no tool execution."""

    runtime_kind = "llm_direct"

    async def run(self, item: dict[str, Any]) -> AgentOutput:
        started = time.perf_counter()
        trajectory_id = str(uuid.uuid4())
        recorder = RunRecorder(
            self.run_state_root, experiment_id=self.run_state_root.name,
            runtime_kind=self.runtime_kind, model_profile=self.profile_name,
            sample_id=item["id"], trajectory_id=trajectory_id,
        )
        messages: list[Any] = []
        system = SystemPromptBuilder(self.project_root).build(
            self.config["prompt"]["system"], self.config,
            tool_names=[tool["name"] for tool in item["tools"]],
        )
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(deepcopy(item["turns"][0]))
        tools = deepcopy(item["tools"])
        permissions = item.get("environment", {}).get("_query_permissions") or {}
        if permissions and not permissions.get("allow_tools", True):
            tools = []
        allowed_names = permissions.get("allowed_tool_names")
        if allowed_names is not None:
            tools = [tool for tool in tools if tool["name"] in set(allowed_names)]
        bound = self.model.bind_tools([_openai_tool(tool) for tool in tools]) if tools else self.model
        retry_events: list[dict[str, Any]] = []
        retry_config = self.config.get("retry", {})
        timeout_policy = retry_config.get("timeout", {"enabled": True, "max_retries": 2})
        transport_policy = retry_config.get("transport", {"enabled": True, "max_retries": 2})
        max_retries = max(int(timeout_policy["max_retries"]), int(transport_policy["max_retries"]))
        timeout = float(self.config["agent"]["sample_timeout_seconds"])

        for attempt in range(1, max_retries + 2):
            recorder.event("model_request", {"messages": recorder.bounded_payload(messages), "tools": tools}, attempt_id=attempt)
            try:
                response = await asyncio.wait_for(bound.ainvoke(messages), timeout=timeout)
                serialized = _serialize_message(response)
                usage = _response_usage(serialized)
                calls = _tool_calls(response, serialized)
                text = _content_text(serialized.get("content"))
                recorder.event("model_response", {"response": serialized, "usage": usage,
                               "tool_calls": calls, "stop_reason": _done_reason(serialized)}, attempt_id=attempt)
                status = "limit_exceeded" if _done_reason(serialized) == "length" else "success"
                recorder.checkpoint({"messages": [*messages, serialized], "final_answer": text,
                                     "tool_calls": calls, "status": status, "usage": usage},
                                    kind="finish", attempt_id=attempt)
                return AgentOutput(
                    id=item["id"], status=status, final_answer=text, messages=[serialized],
                    tool_calls=calls, usage=usage,
                    timing={"duration_seconds": time.perf_counter() - started},
                    error=None if status == "success" else {
                        "type": "output_token_limit", "message": "model output reached max_tokens"},
                    metadata={"runtime": self.runtime_kind, "profile": self.profile_name,
                              "provider": self.profile.get("provider", "openai_compatible"),
                              "exact_model_name": self.profile["name"], "trajectory_id": trajectory_id,
                              "attempt_count": attempt, "retry_events": retry_events,
                              "tool_execution": False},
                )
            except asyncio.TimeoutError:
                kind, message, retryable = "timeout", "sample timeout exceeded", bool(timeout_policy["enabled"])
                allowed_retries = int(timeout_policy["max_retries"])
            except Exception as exc:
                kind, message = type(exc).__name__, str(exc)
                retryable = bool(transport_policy["enabled"]) and _is_retryable_transport_error(exc)
                allowed_retries = int(transport_policy["max_retries"])
            retry_events.append({"attempt": attempt, "error_type": kind, "message": message})
            recorder.event("retry" if retryable and attempt <= allowed_retries else "failure",
                           retry_events[-1], attempt_id=attempt)
            if not retryable or attempt > allowed_retries:
                return AgentOutput(
                    id=item["id"], status="timeout" if kind == "timeout" else "runtime_error",
                    final_answer=None, timing={"duration_seconds": time.perf_counter() - started},
                    error={"type": kind, "message": message},
                    metadata={"runtime": self.runtime_kind, "profile": self.profile_name,
                              "provider": self.profile.get("provider", "openai_compatible"),
                              "exact_model_name": self.profile["name"], "trajectory_id": trajectory_id,
                              "attempt_count": attempt, "retry_events": retry_events},
                )
            await _sleep_before_retry(attempt)
        raise AssertionError("unreachable")
