from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from Agent.evaluation.models import AgentOutput


@dataclass(frozen=True)
class Attachment:
    name: str
    content: str
    media_type: str = "text/plain"


@dataclass(frozen=True)
class PermissionPolicy:
    allow_tools: bool = True
    allowed_tool_names: tuple[str, ...] | None = None
    allow_tool_execution: bool = True


@dataclass(frozen=True)
class CostPolicy:
    max_session_tokens: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    max_session_cost: float | None = None


@dataclass(frozen=True)
class QueryOptions:
    session_id: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    persist_session: bool = True
    attachments: tuple[Attachment, ...] = ()
    permissions: PermissionPolicy | None = None
    cost: CostPolicy = field(default_factory=CostPolicy)
    max_stop_hook_retries: int = 1
    event_sink: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None


class BeforeQueryHook(Protocol):
    def __call__(self, item: dict[str, Any], options: QueryOptions) -> dict[str, Any] | None | Awaitable[dict[str, Any] | None]: ...


class AfterQueryHook(Protocol):
    def __call__(self, output: AgentOutput, options: QueryOptions) -> None | Awaitable[None]: ...


class StopHook(Protocol):
    def __call__(self, output: AgentOutput, options: QueryOptions) -> bool | str | None | Awaitable[bool | str | None]: ...


@dataclass
class QueryHooks:
    before_query: BeforeQueryHook | None = None
    after_query: AfterQueryHook | None = None
    stop: StopHook | None = None


@dataclass
class SessionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class QueryBudgetExceeded(RuntimeError):
    pass


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.jsonl"

    def append(self, session_id: str, event: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {"session_id": session_id, "timestamp": time.time(), **event}
        with self._path(session_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


class QueryEngine:
    """Outer lifecycle around a runtime; benchmark defaults are intentionally inert."""

    def __init__(self, runtime: Any, *, state_root: Path, hooks: QueryHooks | None = None) -> None:
        self.runtime = runtime
        self.store = SessionStore(state_root / "sessions")
        self.hooks = hooks or QueryHooks()
        self._locks: dict[str, asyncio.Lock] = {}
        self._completed: dict[tuple[str, str], AgentOutput] = {}
        self._usage: dict[str, SessionUsage] = {}

    async def submit(self, item: dict[str, Any], options: QueryOptions) -> AgentOutput:
        lock = self._locks.setdefault(options.session_id, asyncio.Lock())
        async with lock:
            key = (options.session_id, options.message_id)
            if key in self._completed:
                if options.persist_session:
                    self.store.append(options.session_id, {"event": "duplicate_message", "message_id": options.message_id})
                return self._completed[key]
            self._check_budget(options)
            prepared = deepcopy(item)
            if options.attachments:
                self._inject_attachments(prepared, options.attachments)
            if options.permissions is not None:
                prepared.setdefault("environment", {})["_query_permissions"] = asdict(options.permissions)
            if options.event_sink is not None:
                prepared.setdefault("environment", {})["_query_event_sink"] = options.event_sink
            if self.hooks.before_query:
                replacement = await _maybe_await(self.hooks.before_query(prepared, options))
                if replacement is not None:
                    prepared = replacement
            if options.persist_session:
                self.store.append(options.session_id, {
                    "event": "query_started", "message_id": options.message_id,
                    "sample_id": prepared.get("id"), "attachments": [asdict(value) for value in options.attachments],
                })

            stop_retries = 0
            while True:
                output = await self.runtime.run(prepared)
                self._accumulate_usage(options.session_id, output, options.cost)
                if self.hooks.after_query:
                    await _maybe_await(self.hooks.after_query(output, options))
                stop_decision = await _maybe_await(self.hooks.stop(output, options)) if self.hooks.stop else True
                if stop_decision is not False and not isinstance(stop_decision, str):
                    break
                if stop_retries >= options.max_stop_hook_retries:
                    if options.persist_session:
                        self.store.append(options.session_id, {
                            "event": "stop_hook_exhausted", "message_id": options.message_id,
                        })
                    break
                feedback = stop_decision if isinstance(stop_decision, str) else "The stop hook blocked completion. Continue and address the missing requirement."
                self._append_stop_feedback(prepared, output, feedback)
                stop_retries += 1
                if options.persist_session:
                    self.store.append(options.session_id, {
                        "event": "stop_hook_blocked", "message_id": options.message_id,
                        "retry": stop_retries, "feedback": feedback,
                    })

            output.metadata["query_engine"] = {
                "session_id": options.session_id, "message_id": options.message_id,
                "stop_hook_retries": stop_retries,
                "session_usage": asdict(self._usage[options.session_id]),
            }
            self._completed[key] = output
            if options.persist_session:
                self.store.append(options.session_id, {
                    "event": "query_completed", "message_id": options.message_id,
                    "output": output.to_dict(),
                })
            return output

    async def stream(self, item: dict[str, Any], options: QueryOptions) -> AsyncIterator[dict[str, Any]]:
        """Yield lifecycle/tool events and finish with a query_completed event."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        original_sink = options.event_sink

        async def sink(event: dict[str, Any]) -> None:
            await queue.put(event)
            if original_sink is not None:
                await _maybe_await(original_sink(event))

        streamed_options = replace(options, event_sink=sink)
        task = asyncio.create_task(self.submit(item, streamed_options))
        yield {"event": "query_started", "session_id": options.session_id, "message_id": options.message_id}
        while not task.done() or not queue.empty():
            if task.done() and queue.empty():
                break
            try:
                yield await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
        output = await task
        yield {"event": "query_completed", "output": output}

    def session_usage(self, session_id: str) -> SessionUsage:
        return deepcopy(self._usage.get(session_id, SessionUsage()))

    def _check_budget(self, options: QueryOptions) -> None:
        usage = self._usage.get(options.session_id, SessionUsage())
        if options.cost.max_session_tokens is not None and usage.total_tokens >= options.cost.max_session_tokens:
            raise QueryBudgetExceeded("session token budget exhausted")
        if options.cost.max_session_cost is not None and usage.estimated_cost >= options.cost.max_session_cost:
            raise QueryBudgetExceeded("session cost budget exhausted")

    def _accumulate_usage(self, session_id: str, output: AgentOutput, policy: CostPolicy) -> None:
        usage = self._usage.setdefault(session_id, SessionUsage())
        prompt = int((output.usage or {}).get("prompt_tokens") or 0)
        completion = int((output.usage or {}).get("completion_tokens") or 0)
        total = int((output.usage or {}).get("total_tokens") or prompt + completion)
        usage.prompt_tokens += prompt
        usage.completion_tokens += completion
        usage.total_tokens += total
        usage.estimated_cost += (
            prompt * float(policy.input_cost_per_million or 0) / 1_000_000
            + completion * float(policy.output_cost_per_million or 0) / 1_000_000
        )

    @staticmethod
    def _inject_attachments(item: dict[str, Any], attachments: tuple[Attachment, ...]) -> None:
        turns = item.setdefault("turns", [[]])
        if not turns:
            turns.append([])
        blocks = "\n\n".join(
            f'<attachment name="{value.name}" media_type="{value.media_type}">\n{value.content}\n</attachment>'
            for value in attachments
        )
        user = next((message for message in turns[0] if message.get("role") == "user"), None)
        if user is None:
            turns[0].append({"role": "user", "content": blocks})
        else:
            user["content"] = str(user.get("content", "")) + "\n\n" + blocks

    @staticmethod
    def _append_stop_feedback(item: dict[str, Any], output: AgentOutput, feedback: str) -> None:
        turns = item.setdefault("turns", [[]])
        if not turns:
            turns.append([])
        if output.final_answer:
            turns[0].append({"role": "assistant", "content": output.final_answer})
        turns[0].append({"role": "user", "content": feedback})
