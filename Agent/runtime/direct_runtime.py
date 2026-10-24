from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.evaluation.models import AgentOutput
from Agent.runtime.bfcl_executor import BFCLProtocolExecutor
from Agent.telemetry import RunRecorder, normalized_token_count
from Agent.prompting import SystemPromptBuilder


logger = logging.getLogger(__name__)
MAX_TRANSIENT_RETRIES = 2


class DirectLangChainRuntime:
    """Shared V3 model/tool loop used by protocol and memory runtimes."""

    def __init__(self, config: dict[str, Any], project_root: Path, profile_name: str | None = None, model: Any | None = None) -> None:
        self.config = config
        self.project_root = project_root
        self.profile_name = profile_name or config["model"]["default_profile"]
        if self.profile_name not in config["model"]["profiles"]:
            raise ValueError(f"unknown model profile: {self.profile_name}")
        self.profile = config["model"]["profiles"][self.profile_name]
        self.run_state_root = project_root / "runs" / f"{config['dataset']['name']}-{self.profile_name}"
        self.model = model if model is not None else self._create_model()
        self.bfcl_executor = BFCLProtocolExecutor(project_root)

    def _create_model(self) -> Any:
        provider = self.profile.get("provider", "openai_compatible")
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise RuntimeError("Ollama profile requires langchain-ollama") from exc
            host = os.environ.get(self.profile.get("host_env", "OLLAMA_HOST"), self.profile.get("host", "http://127.0.0.1:11434"))
            kwargs: dict[str, Any] = {
                "model": self.profile["name"], "base_url": host,
                "temperature": self.profile["temperature"],
                "num_ctx": self.profile["context_window"],
                "num_predict": self.profile["max_tokens"],
            }
            reasoning = self.config["model"]["reasoning"]
            # Ollama models such as Qwen may enable thinking by default when the
            # option is omitted, so always send the explicit boolean value.
            kwargs["reasoning"] = bool(reasoning["enabled"])
            if "seed" in self.profile:
                kwargs["seed"] = int(self.profile["seed"])
            return ChatOllama(**kwargs)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("runtime requires langchain-openai") from exc
        base_url = _required_env(self.profile["base_url_env"])
        api_key = _required_env(self.profile["api_key_env"])
        kwargs: dict[str, Any] = {
            "model": self.profile["name"], "base_url": base_url, "api_key": api_key,
            "temperature": self.profile["temperature"], "max_tokens": self.profile["max_tokens"], "max_retries": 0,
        }
        reasoning = self.config["model"]["reasoning"]
        if "seed" in self.profile:
            kwargs["seed"] = int(self.profile["seed"])
        if reasoning["enabled"]:
            kwargs["reasoning_effort"] = reasoning["mode"]
        else:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return ChatOpenAI(**kwargs)

    async def run(self, item: dict[str, Any]) -> AgentOutput:
        started = time.perf_counter()
        retry_events: list[dict[str, Any]] = []
        retry_config = self.config.get("retry", {})
        timeout_policy = retry_config.get("timeout", {"enabled": True, "max_retries": MAX_TRANSIENT_RETRIES})
        transport_policy = retry_config.get("transport", {"enabled": True, "max_retries": MAX_TRANSIENT_RETRIES})
        max_attempts = max(int(timeout_policy["max_retries"]), int(transport_policy["max_retries"])) + 2
        for attempt in range(1, max_attempts):
            try:
                output = await asyncio.wait_for(
                    self._run(item, started, attempt),
                    timeout=self.config["agent"]["sample_timeout_seconds"],
                )
                output.metadata["attempt_count"] = attempt
                output.metadata["retry_events"] = retry_events
                return output
            except asyncio.TimeoutError:
                kind = "timeout"
                message = "sample timeout exceeded"
                retryable = bool(timeout_policy["enabled"])
                allowed_retries = int(timeout_policy["max_retries"])
            except Exception as exc:
                kind = type(exc).__name__
                message = str(exc)
                retryable = bool(transport_policy["enabled"]) and _is_retryable_transport_error(exc)
                allowed_retries = int(transport_policy["max_retries"])

            retry_events.append({"attempt": attempt, "error_type": kind, "message": message})
            if not retryable or attempt > allowed_retries:
                return self._failure(item["id"], started, kind, message, attempt, retry_events)

            delay_seconds = attempt
            logger.warning(
                "Retrying sample %s after transient failure (%s), attempt %d/%d in %ds: %s",
                item["id"], kind, attempt + 1, allowed_retries + 1, delay_seconds, message,
            )
            await _sleep_before_retry(delay_seconds)

        raise AssertionError("unreachable")

    async def _run(self, item: dict[str, Any], started: float, attempt: int = 1) -> AgentOutput:
        if item["metadata"]["benchmark"] == "locomo":
            return await self._run_locomo(item, started, attempt)
        messages: list[Any] = []
        system_prompt = SystemPromptBuilder(self.project_root).build(
            self.config["prompt"]["system"], self.config, tool_names=[tool["name"] for tool in item["tools"]]
        )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        tools = deepcopy(item["tools"])
        permissions = item.get("environment", {}).get("_query_permissions") or {}
        if permissions and not permissions.get("allow_tools", True):
            tools = []
        allowed_names = permissions.get("allowed_tool_names")
        if allowed_names is not None:
            tools = [tool for tool in tools if tool["name"] in set(allowed_names)]
        calls_by_turn: list[list[list[str]]] = []
        serialized: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        final_text: str | None = None
        xml_retry_count = 0
        usage = _empty_usage()
        trajectory_id = str(uuid.uuid4())
        recorder = RunRecorder(
            self.run_state_root, experiment_id=self.run_state_root.name,
            runtime_kind=getattr(self, "runtime_kind", "direct_langchain"), model_profile=self.profile_name,
            sample_id=item["id"], trajectory_id=trajectory_id,
        )

        for turn_index, turn in enumerate(item["turns"]):
            additions = item["environment"].get("missed_function", {}).get(str(turn_index), [])
            if additions:
                tools.extend(additions)
                turn = turn or [{"role": "user", "content": "I have updated some more functions you can choose from. What about now?"}]
            messages.extend(turn)
            turn_calls: list[list[str]] = []
            is_agentic = item["metadata"].get("category", "").startswith("web_search")
            step_limit = self.config["agent"]["max_steps"] if item["task_family"] == "multiturn" or is_agentic else 1
            for _ in range(step_limit):
                bound = self.model.bind_tools([_openai_tool(tool) for tool in tools]) if tools else self.model
                if recorder:
                    recorder.event("model_request", {"messages": recorder.bounded_payload(messages), "tools": tools}, attempt_id=attempt)
                response, retries = await _invoke_with_xml_retry(bound, messages, _malformed_retry_limit(self.config))
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                _accumulate_usage(usage, response_dict)
                if recorder:
                    recorder.event("model_response", {"response": response_dict, "usage": _response_usage(response_dict)}, attempt_id=attempt)
                serialized.append(response_dict)
                messages.append(response)
                final_text = _content_text(response_dict.get("content"))
                if _done_reason(response_dict) == "length":
                    return AgentOutput(
                        id=item["id"], status="limit_exceeded", final_answer=final_text,
                        messages=serialized, tool_calls=all_calls,
                        usage=usage,
                        timing={"duration_seconds": time.perf_counter() - started},
                        error={"type": "output_token_limit", "message": "model output reached max_tokens before completion"},
                        metadata={"runtime": "direct_langchain", "profile": self.profile_name,
                                  "bfcl_calls_by_turn": calls_by_turn, "done_reason": "length",
                                  "xml_retry_count": xml_retry_count},
                    )
                response_calls = _tool_calls(response, response_dict)
                if not response_calls:
                    break
                call_strings = [_call_string(call) for call in response_calls]
                turn_calls.append(call_strings)
                all_calls.extend(response_calls)
                if self.config["tool"]["execution"] == "disabled" or not permissions.get("allow_tool_execution", True):
                    break
                results = self._execute_bfcl(item, response_calls, call_strings, attempt)
                for call, result in zip(response_calls, results):
                    if recorder:
                        success = True
                        try:
                            decoded_result = json.loads(result)
                            success = not (isinstance(decoded_result, dict) and "error" in decoded_result)
                        except (ValueError, TypeError):
                            pass
                        payload: Any = {"output": result, "success": success}
                        if len(result) > 2000:
                            payload = {"artifact": recorder.artifact({"tool": call["name"], "output": result}),
                                       "preview": result[:1000], "success": success}
                        event_type = "search_result" if call["name"] in {"search_engine_query", "fetch_url_content"} else "tool_result"
                        event_payload = {"call": call, "result": payload}
                        if event_type == "search_result":
                            event_payload["capture"] = {
                                "mode": self.config.get("features", {}).get("web_search", {}).get("mode", "live"),
                                "backend": os.getenv("BFCL_SEARCH_BACKEND", "ddgs"),
                                "collected_at": datetime.now(timezone.utc).isoformat(),
                            }
                        if call["name"] == "search_engine_query":
                            try:
                                search_rows = json.loads(result)
                                event_payload["ranked_results"] = [
                                    {"rank": index + 1, "url": row.get("href"), "title": row.get("title"),
                                     "snippet": row.get("body")}
                                    for index, row in enumerate(search_rows) if isinstance(row, dict)
                                ]
                            except (ValueError, TypeError):
                                event_payload["ranked_results"] = []
                        recorder.event(event_type, event_payload, attempt_id=attempt)
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)
            calls_by_turn.append(turn_calls)

        self.bfcl_executor.flush_memory_prerequisite(item=item, attempt=attempt)
        if recorder:
            recorder.checkpoint({"messages": serialized, "final_answer": final_text, "tool_calls": all_calls,
                                 "usage": usage, "status": "success"}, kind="finish", attempt_id=attempt)

        return AgentOutput(
            id=item["id"], status="success", final_answer=final_text, messages=serialized,
            tool_calls=all_calls, timing={"duration_seconds": time.perf_counter() - started},
            usage=usage,
            metadata={"runtime": "direct_langchain", "profile": self.profile_name,
                      "bfcl_calls_by_turn": calls_by_turn, "xml_retry_count": xml_retry_count,
                      "trajectory_id": trajectory_id},
        )

    async def _run_locomo(self, item: dict[str, Any], started: float, attempt: int) -> AgentOutput:
        trajectory_id = str(uuid.uuid4())
        recorder = RunRecorder(
            self.run_state_root, experiment_id=self.run_state_root.name,
            runtime_kind=getattr(self, "runtime_kind", "direct_langchain"),
            model_profile=self.profile_name, sample_id=item["id"], trajectory_id=trajectory_id,
        )
        state_dir = self.run_state_root / "locomo_memory"
        state_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", item["id"])
        attempt_db = state_dir / f"{safe_id}.attempt-{attempt}.sqlite3"
        final_db = state_dir / f"{safe_id}.sqlite3"
        if attempt_db.exists():
            attempt_db.unlink()
        _initialize_memory_db(attempt_db)

        serialized: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        xml_retry_count = 0
        usage = _empty_usage()
        base_prompt = self.config["prompt"]["system"]
        ingestion_prompt = (
            base_prompt + "\n\nYou are ingesting one conversation session. Use memory_add, memory_update, "
            "or memory_delete as needed. Do not answer the conversation participants. Finish only after durable "
            "information from this session has been stored."
        )

        for session_index, turn in enumerate(item["turns"]):
            messages: list[Any] = [{"role": "system", "content": ingestion_prompt}, *deepcopy(turn)]
            for _ in range(self.config["agent"]["max_steps"]):
                bound = self.model.bind_tools([_openai_tool(tool) for tool in item["tools"]])
                if recorder:
                    recorder.event("model_request", {"phase": "memory_ingestion", "session_index": session_index,
                                                       "messages": recorder.bounded_payload(messages)}, attempt_id=attempt)
                response, retries = await _invoke_with_xml_retry(bound, messages, _malformed_retry_limit(self.config))
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                _accumulate_usage(usage, response_dict)
                serialized.append(response_dict)
                messages.append(response)
                if recorder:
                    recorder.event("model_response", {"phase": "memory_ingestion", "session_index": session_index,
                                                        "response": response_dict, "usage": _response_usage(response_dict)},
                                   attempt_id=attempt)
                if _done_reason(response_dict) == "length":
                    return _limit_output(item, started, serialized, all_calls, xml_retry_count)
                calls = _tool_calls(response, response_dict)
                if not calls:
                    break
                for call in calls:
                    call["phase"] = "ingestion"
                    call["session_index"] = session_index
                    result = _execute_locomo_memory(attempt_db, call, read_only=False)
                    all_calls.append(call)
                    if recorder:
                        recorder.event("tool_result", {"phase": "memory_ingestion", "session_index": session_index,
                                                        "call": call, "result": result}, attempt_id=attempt)
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)
            if recorder:
                recorder.checkpoint({"phase": "memory_ingestion", "session_index": session_index,
                                     "usage": usage, "tool_calls": all_calls}, kind="tool_batch", attempt_id=attempt)

        search_tools = [tool for tool in item["tools"] if tool["name"] == "memory_search"]
        question_prompt = (
            base_prompt + "\n\nYou are answering a held-out question. The original conversation is not in "
            "your context. Search durable memory before answering. Do not add, modify, or delete memories. "
            "Return only the short answer."
        )
        for question in item["environment"]["questions"]:
            messages = [
                {"role": "system", "content": question_prompt},
                {"role": "user", "content": question["question"]},
            ]
            question_calls: list[dict[str, Any]] = []
            final_text: str | None = None
            for _ in range(self.config["agent"]["max_steps"]):
                bound = self.model.bind_tools([_openai_tool(tool) for tool in search_tools])
                if recorder:
                    recorder.event("model_request", {"phase": "memory_qa", "question_id": question["id"],
                                                       "messages": recorder.bounded_payload(messages)}, attempt_id=attempt)
                response, retries = await _invoke_with_xml_retry(bound, messages, _malformed_retry_limit(self.config))
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                _accumulate_usage(usage, response_dict)
                serialized.append(response_dict)
                messages.append(response)
                if recorder:
                    recorder.event("model_response", {"phase": "memory_qa", "question_id": question["id"],
                                                        "response": response_dict, "usage": _response_usage(response_dict)},
                                   attempt_id=attempt)
                final_text = _content_text(response_dict.get("content"))
                if _done_reason(response_dict) == "length":
                    return _limit_output(item, started, serialized, all_calls, xml_retry_count)
                calls = _tool_calls(response, response_dict)
                if not calls:
                    break
                for call in calls:
                    call["phase"] = "question"
                    call["question_id"] = question["id"]
                    result = _execute_locomo_memory(attempt_db, call, read_only=True)
                    question_calls.append(call)
                    all_calls.append(call)
                    if recorder:
                        recorder.event("tool_result", {"phase": "memory_qa", "question_id": question["id"],
                                                        "call": call, "result": result}, attempt_id=attempt)
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)
            answers.append({"id": question["id"], "category": question["category"],
                            "answer": final_text or "", "tool_calls": question_calls})

        os.replace(attempt_db, final_db)
        if recorder:
            recorder.checkpoint({"phase": "finished", "answers": answers, "usage": usage,
                                 "tool_calls": all_calls}, kind="finish", attempt_id=attempt)
        return AgentOutput(
            id=item["id"], status="success", final_answer=answers[-1]["answer"] if answers else None,
            messages=serialized, tool_calls=all_calls,
            usage=usage,
            timing={"duration_seconds": time.perf_counter() - started},
            metadata={"runtime": "direct_langchain", "profile": self.profile_name,
                      "locomo_answers": answers, "memory_db": str(final_db),
                      "xml_retry_count": xml_retry_count, "trajectory_id": trajectory_id,
                      "locomo_mode": "persistent_memory"},
        )

    def _execute_bfcl(
        self, item: dict[str, Any], response_calls: list[dict[str, Any]],
        calls: list[str], attempt: int,
    ) -> list[str]:
        results = self.bfcl_executor.execute_batch(
            calls, item=item, trajectory_id=f"{item['id']}:attempt-{attempt}",
            tool_call_ids=[call["id"] for call in response_calls], attempt=attempt,
        )
        return [result.output for result in results]

    @staticmethod
    def _failure(
        sample_id: str, started: float, kind: str, message: str,
        attempt_count: int = 1, retry_events: list[dict[str, Any]] | None = None,
    ) -> AgentOutput:
        return AgentOutput(sample_id, "timeout" if kind == "timeout" else "runtime_error", None,
                           timing={"duration_seconds": time.perf_counter() - started},
                           error={"type": kind, "message": message},
                           metadata={"runtime": "direct_langchain", "attempt_count": attempt_count,
                                     "retry_events": retry_events or []})


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable is not set: {name}")
    return value


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    # BFCL records also contain a non-standard ``response`` schema.  Native
    # tool APIs accept only the function name, description, and input schema;
    # forwarding the response schema makes some Ollama parsers unstable.
    function = {
        "name": str(tool["name"]),
        "description": str(tool.get("description", "")),
        "parameters": _fix_schema(tool.get("parameters", {"type": "object", "properties": {}})),
    }
    return {"type": "function", "function": function}


def _fix_schema(value: Any) -> Any:
    if isinstance(value, dict):
        type_names = {
            "dict": "object", "float": "number", "str": "string",
            "bool": "boolean", "list": "array",
        }
        return {
            key: (type_names.get(child, child) if key == "type" and isinstance(child, str) else _fix_schema(child))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_fix_schema(child) for child in value]
    return value


def _serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return deepcopy(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return {"role": "assistant", "content": getattr(message, "content", str(message))}


def _tool_calls(message: Any, value: dict[str, Any]) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or value.get("tool_calls") or []
    normalized = []
    for call in calls:
        function = call.get("function", {})
        args = call.get("args", function.get("arguments", {}))
        if isinstance(args, str):
            args = json.loads(args)
        normalized.append({"id": call.get("id") or str(uuid.uuid4()), "name": call.get("name") or function.get("name"), "args": args})
    return normalized


def _call_string(call: dict[str, Any]) -> str:
    arguments = ", ".join(f"{key}={value!r}" for key, value in call["args"].items())
    return f"{call['name']}({arguments})"


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return text or None
    return str(content) if content is not None else None


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_reasoning_tokens": 0,
        "visible_step_tokens": 0,
        "normalized_completion_tokens": 0,
    }


def _response_usage(message: dict[str, Any]) -> dict[str, int]:
    raw = message.get("usage_metadata") or {}
    response_metadata = message.get("response_metadata") or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    prompt = raw.get("input_tokens", token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))) or 0
    completion = raw.get("output_tokens", token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))) or 0
    total = raw.get("total_tokens", token_usage.get("total_tokens", int(prompt) + int(completion))) or 0
    details = raw.get("output_token_details") or token_usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning", details.get("reasoning_tokens", 0)) or 0
    text = _content_text(message.get("content")) or ""
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "provider_reasoning_tokens": int(reasoning),
        "visible_step_tokens": normalized_token_count(text),
        "normalized_completion_tokens": normalized_token_count(text),
    }


def _accumulate_usage(total: dict[str, int], message: dict[str, Any]) -> None:
    for key, value in _response_usage(message).items():
        total[key] = total.get(key, 0) + value


def _done_reason(message: dict[str, Any]) -> str | None:
    metadata = message.get("response_metadata") or {}
    value = metadata.get("done_reason") or metadata.get("finish_reason")
    return str(value) if value is not None else None


def _malformed_retry_limit(config: dict[str, Any]) -> int:
    policy = config.get("retry", {}).get("malformed_response")
    if policy is None:
        return 2
    return int(policy["max_retries"]) if policy.get("enabled") else 0


async def _invoke_with_xml_retry(model: Any, messages: list[Any], max_retries: int = 2) -> tuple[Any, int]:
    """Retry malformed responses before the current step executes any tool."""
    retries = 0
    request_messages = list(messages)
    while True:
        try:
            return await model.ainvoke(request_messages), retries
        except Exception as exc:
            if retries >= max_retries or not _is_retryable_xml_response_error(exc):
                raise
            retries += 1
            request_messages = [*messages, {
                "role": "system",
                "content": (
                    "Your previous tool call could not be parsed. Return a native structured tool call. "
                    "Every tool call must identify the function before supplying its parameters; never emit "
                    "a parameter element outside its function element. Do not include explanatory XML."
                ),
            }]


def _is_retryable_xml_response_error(exc: Exception) -> bool:
    if type(exc).__name__ != "ResponseError":
        return False
    message = str(exc).lower()
    return any(marker in message for marker in (
        "xml syntax error",
        "expected element type <function>",
        "expected element type <parameter>",
    ))


def _is_retryable_transport_error(exc: Exception) -> bool:
    # Deterministic model-output format errors are handled by the inner retry
    # with corrective context. Replaying the entire sample as a transport
    # retry only repeats already executed turns and the same malformed output.
    if _is_retryable_xml_response_error(exc):
        return False
    if type(exc).__name__ in {
        "OpenAIConnectionError", "APIConnectionError", "ConnectError",
        "ConnectionError", "ReadError", "RemoteProtocolError",
    }:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "502 bad gateway" in message or "internal server error" in message


async def _sleep_before_retry(delay_seconds: int) -> None:
    await asyncio.sleep(delay_seconds)


def _initialize_memory_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                date TEXT NOT NULL DEFAULT '',
                source_ids TEXT NOT NULL DEFAULT '',
                UNIQUE(content, date, source_ids)
            );
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                content, date, source_ids, content='memories', content_rowid='id'
            );
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, date, source_ids)
                VALUES (new.id, new.content, new.date, new.source_ids);
            END;
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, date, source_ids)
                VALUES ('delete', old.id, old.content, old.date, old.source_ids);
            END;
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, date, source_ids)
                VALUES ('delete', old.id, old.content, old.date, old.source_ids);
                INSERT INTO memories_fts(rowid, content, date, source_ids)
                VALUES (new.id, new.content, new.date, new.source_ids);
            END;
        """)
        connection.commit()
    finally:
        connection.close()


def _execute_locomo_memory(path: Path, call: dict[str, Any], *, read_only: bool) -> str:
    name = call["name"]
    args = call["args"]
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        if name == "memory_search":
            limit = max(1, min(int(args.get("limit", 5)), 20))
            tokens = re.findall(r"\w+", str(args.get("query", "")).lower(), flags=re.UNICODE)
            if tokens:
                expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"*' for token in tokens)
                rows = connection.execute(
                    "SELECT m.id, m.content, m.date, m.source_ids, bm25(memories_fts) AS rank "
                    "FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?", (expression, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, content, date, source_ids FROM memories ORDER BY id DESC LIMIT ?", (limit,),
                ).fetchall()
            return json.dumps([dict(row) for row in rows], ensure_ascii=False)
        if read_only:
            return json.dumps({"error": "only memory_search is allowed during evaluation"})
        source_ids = json.dumps(args.get("source_ids", []), ensure_ascii=False)
        if name == "memory_add":
            connection.execute(
                "INSERT OR IGNORE INTO memories(content, date, source_ids) VALUES (?, ?, ?)",
                (str(args["content"]), str(args.get("date", "")), source_ids),
            )
            row = connection.execute(
                "SELECT id FROM memories WHERE content=? AND date=? AND source_ids=?",
                (str(args["content"]), str(args.get("date", "")), source_ids),
            ).fetchone()
            connection.commit()
            return json.dumps({"status": "stored", "memory_id": row["id"]})
        if name == "memory_update":
            cursor = connection.execute(
                "UPDATE memories SET content=?, date=?, source_ids=? WHERE id=?",
                (str(args["content"]), str(args.get("date", "")), source_ids, int(args["memory_id"])),
            )
            connection.commit()
            return json.dumps({"status": "updated", "matched": cursor.rowcount})
        if name == "memory_delete":
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (int(args["memory_id"]),))
            connection.commit()
            return json.dumps({"status": "deleted", "matched": cursor.rowcount})
        return json.dumps({"error": f"unknown memory tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False)
    finally:
        if connection is not None:
            connection.close()


def _limit_output(
    item: dict[str, Any], started: float, messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]], xml_retry_count: int,
) -> AgentOutput:
    return AgentOutput(
        id=item["id"], status="limit_exceeded", final_answer=None, messages=messages, tool_calls=tool_calls,
        timing={"duration_seconds": time.perf_counter() - started},
        error={"type": "output_token_limit", "message": "model output reached max_tokens before completion"},
        metadata={"runtime": "direct_langchain", "done_reason": "length",
                  "xml_retry_count": xml_retry_count},
    )
