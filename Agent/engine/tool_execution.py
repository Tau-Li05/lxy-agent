from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ParallelToolResult:
    index: int
    call: dict[str, Any]
    value: Any
    duration_seconds: float


EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _emit(sink: EventSink | None, event: dict[str, Any]) -> None:
    if sink is None:
        return
    result = sink(event)
    if hasattr(result, "__await__"):
        await result


async def execute_parallel(
    calls: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Awaitable[Any]],
    *,
    max_concurrency: int = 4,
    event_sink: EventSink | None = None,
) -> list[ParallelToolResult]:
    """Execute independent calls concurrently and return them in request order.

    Completion events are emitted immediately, while the returned list is stable.
    A worker exception is retained as a result so one failed call cannot cancel its
    siblings; the runtime decides how to serialize that error for the model.
    """
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run(index: int, call: dict[str, Any]) -> ParallelToolResult:
        started = time.perf_counter()
        await _emit(event_sink, {
            "event": "tool_started", "index": index,
            "tool_call_id": call.get("id"), "name": call.get("name"),
        })
        try:
            async with semaphore:
                value = await worker(call)
        except Exception as exc:  # A batch should preserve every call result.
            value = exc
        duration = time.perf_counter() - started
        await _emit(event_sink, {
            "event": "tool_completed", "index": index,
            "tool_call_id": call.get("id"), "name": call.get("name"),
            "success": not isinstance(value, Exception), "duration_seconds": duration,
        })
        return ParallelToolResult(index, call, value, duration)

    results = await asyncio.gather(*(run(index, call) for index, call in enumerate(calls)))
    return sorted(results, key=lambda result: result.index)
