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
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.evaluation.models import AgentOutput


logger = logging.getLogger(__name__)
MAX_TRANSIENT_RETRIES = 2


class DirectLangChainRuntime:
    """Runs v2 records verbatim; dataset conversion is intentionally absent here."""

    def __init__(self, config: dict[str, Any], project_root: Path, profile_name: str | None = None, model: Any | None = None) -> None:
        self.config = config
        self.project_root = project_root
        self.profile_name = profile_name or config["model"]["default_profile"]
        if self.profile_name not in config["model"]["profiles"]:
            raise ValueError(f"unknown model profile: {self.profile_name}")
        self.profile = config["model"]["profiles"][self.profile_name]
        self.run_state_root = project_root / "runs" / f"{config['dataset']['name']}-{self.profile_name}"
        self.model = model if model is not None else self._create_model()

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
        for attempt in range(1, MAX_TRANSIENT_RETRIES + 2):
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
                retryable = True
            except Exception as exc:
                kind = type(exc).__name__
                message = str(exc)
                retryable = _is_retryable_transport_error(exc)

            retry_events.append({"attempt": attempt, "error_type": kind, "message": message})
            if not retryable or attempt > MAX_TRANSIENT_RETRIES:
                return self._failure(item["id"], started, kind, message, attempt, retry_events)

            delay_seconds = attempt
            logger.warning(
                "Retrying sample %s after transient failure (%s), attempt %d/%d in %ds: %s",
                item["id"], kind, attempt + 1, MAX_TRANSIENT_RETRIES + 1, delay_seconds, message,
            )
            await _sleep_before_retry(delay_seconds)

        raise AssertionError("unreachable")

    async def _run(self, item: dict[str, Any], started: float, attempt: int = 1) -> AgentOutput:
        if item["metadata"]["benchmark"] == "locomo":
            return await self._run_locomo(item, started, attempt)
        messages: list[Any] = []
        if self.config["prompt"]["system"]:
            messages.append({"role": "system", "content": self.config["prompt"]["system"]})
        tools = deepcopy(item["tools"])
        calls_by_turn: list[list[list[str]]] = []
        serialized: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        final_text: str | None = None
        xml_retry_count = 0

        for turn_index, turn in enumerate(item["turns"]):
            additions = item["environment"].get("missed_function", {}).get(str(turn_index), [])
            if additions:
                tools.extend(additions)
                turn = turn or [{"role": "user", "content": "I have updated some more functions you can choose from. What about now?"}]
            messages.extend(turn)
            turn_calls: list[list[str]] = []
            step_limit = 1 if item["task_family"] == "single_turn" else self.config["agent"]["max_steps"]
            for _ in range(step_limit):
                bound = self.model.bind_tools([_openai_tool(tool) for tool in tools]) if tools else self.model
                response, retries = await _invoke_with_xml_retry(bound, messages)
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                serialized.append(response_dict)
                messages.append(response)
                final_text = _content_text(response_dict.get("content"))
                if _done_reason(response_dict) == "length":
                    return AgentOutput(
                        id=item["id"], status="limit_exceeded", final_answer=final_text,
                        messages=serialized, tool_calls=all_calls,
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
                if self.config["tool"]["execution"] == "disabled":
                    break
                results = self._execute_bfcl(item, call_strings, attempt)
                for call, result in zip(response_calls, results):
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)
            calls_by_turn.append(turn_calls)

        return AgentOutput(
            id=item["id"], status="success", final_answer=final_text, messages=serialized,
            tool_calls=all_calls, timing={"duration_seconds": time.perf_counter() - started},
            metadata={"runtime": "direct_langchain", "profile": self.profile_name,
                      "bfcl_calls_by_turn": calls_by_turn, "xml_retry_count": xml_retry_count},
        )

    async def _run_locomo(self, item: dict[str, Any], started: float, attempt: int) -> AgentOutput:
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
                response, retries = await _invoke_with_xml_retry(bound, messages)
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                serialized.append(response_dict)
                messages.append(response)
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
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)

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
                response, retries = await _invoke_with_xml_retry(bound, messages)
                xml_retry_count += retries
                response_dict = _serialize_message(response)
                serialized.append(response_dict)
                messages.append(response)
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
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
                    messages.append(tool_message)
                    serialized.append(tool_message)
            answers.append({"id": question["id"], "category": question["category"],
                            "answer": final_text or "", "tool_calls": question_calls})

        os.replace(attempt_db, final_db)
        return AgentOutput(
            id=item["id"], status="success", final_answer=answers[-1]["answer"] if answers else None,
            messages=serialized, tool_calls=all_calls,
            timing={"duration_seconds": time.perf_counter() - started},
            metadata={"runtime": "direct_langchain", "profile": self.profile_name,
                      "locomo_answers": answers, "memory_db": str(final_db),
                      "xml_retry_count": xml_retry_count},
        )

    def _execute_bfcl(self, item: dict[str, Any], calls: list[str], attempt: int) -> list[str]:
        package = self.project_root / "benchmark/BFCL-main/berkeley-function-call-leaderboard"
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

        environment = item["environment"]
        results, _ = execute_multi_turn_func_call(
            calls, environment["initial_config"], environment["involved_classes"],
            f"direct_{self.profile_name}_attempt_{attempt}", item["id"],
            long_context="long_context" in item["metadata"]["category"], is_evaL_run=False,
        )
        return results

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
    function = deepcopy(tool)
    function["parameters"] = _fix_schema(function.get("parameters", {"type": "object", "properties": {}}))
    return {"type": "function", "function": function}


def _fix_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("object" if key == "type" and child == "dict" else _fix_schema(child)) for key, child in value.items()}
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


def _done_reason(message: dict[str, Any]) -> str | None:
    metadata = message.get("response_metadata") or {}
    value = metadata.get("done_reason") or metadata.get("finish_reason")
    return str(value) if value is not None else None


async def _invoke_with_xml_retry(model: Any, messages: list[Any]) -> tuple[Any, int]:
    """Retry malformed responses before the current step executes any tool."""
    retries = 0
    while True:
        try:
            return await model.ainvoke(messages), retries
        except Exception as exc:
            if retries >= 2 or not _is_retryable_xml_response_error(exc):
                raise
            retries += 1


def _is_retryable_xml_response_error(exc: Exception) -> bool:
    return type(exc).__name__ == "ResponseError" and "XML syntax error" in str(exc)


def _is_retryable_transport_error(exc: Exception) -> bool:
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
