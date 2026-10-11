from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from Agent.evaluation.models import AgentInput, AgentOutput


class LangChainRuntime:
    """LangChain adapter. It deliberately accepts no reference store."""

    def __init__(self, config: dict[str, Any], agent: Any | None = None) -> None:
        self.config = config
        self.limits = config["runtime"]["limits"]
        self._semaphore = asyncio.Semaphore(self.limits["concurrency"])
        self._agent = agent if agent is not None else self._create_agent(config)

    @staticmethod
    def _create_agent(config: dict[str, Any]) -> Any:
        try:
            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("LangChain runtime requires langchain and langchain-openai") from exc

        model_config = config["model"]
        model_name = _required_env(model_config["model_env"])
        base_url = _required_env(model_config["base_url_env"])
        api_key = _required_env(model_config["api_key_env"])
        model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=model_config["temperature"],
            max_retries=model_config["max_retries"],
            max_tokens=model_config["max_output_tokens"],
        )
        return create_agent(model=model, tools=[], system_prompt=config.get("system_prompt"))

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        started = time.perf_counter()
        try:
            if agent_input.tools:
                return self._failure(agent_input.id, "invalid_input", started, "tools_not_allowed", "DEV runtime requires empty tools")
            async with self._semaphore:
                result = await asyncio.wait_for(
                    self._agent.ainvoke(
                        {"messages": agent_input.messages},
                        config={"recursion_limit": self.limits["max_steps"]},
                    ),
                    timeout=self.limits["sample_timeout_seconds"],
                )
            messages = [_serialize_message(message) for message in result.get("messages", [])]
            final = messages[-1].get("content") if messages else None
            if not isinstance(final, str):
                final = _content_to_text(final)
            tool_calls = [call for message in messages for call in message.get("tool_calls", [])]
            usage, available = _collect_usage(result.get("messages", []))
            if tool_calls or len(tool_calls) > self.limits["max_tool_calls"]:
                status = "limit_exceeded"
                error = {"type": "tool_call_limit", "message": "tool calls are disabled"}
            elif not final:
                status = "success"
                error = None
            else:
                status = "success"
                error = None
            return AgentOutput(
                id=agent_input.id,
                status=status,
                final_answer=final,
                messages=messages,
                tool_calls=tool_calls,
                usage=usage,
                timing={"duration_seconds": time.perf_counter() - started},
                error=error,
                metadata={"runtime": "langchain", "usage_available": available},
            )
        except asyncio.TimeoutError:
            return self._failure(agent_input.id, "timeout", started, "timeout", "sample timeout exceeded")
        except Exception as exc:  # model/provider exceptions become per-sample results
            return self._failure(agent_input.id, "runtime_error", started, type(exc).__name__, str(exc))

    @staticmethod
    def _failure(sample_id: str, status: str, started: float, error_type: str, message: str) -> AgentOutput:
        return AgentOutput(
            id=sample_id,
            status=status,
            final_answer=None,
            timing={"duration_seconds": time.perf_counter() - started},
            error={"type": error_type, "message": message},
            metadata={"runtime": "langchain", "usage_available": False},
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable is not set: {name}")
    return value


def _serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        value = dict(message)
    elif hasattr(message, "model_dump"):
        value = message.model_dump(mode="json")
    else:
        value = {"type": type(message).__name__, "content": getattr(message, "content", str(message))}
    value.setdefault("content", "")
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
        return "".join(parts) or None
    return str(content) if content is not None else None


def _collect_usage(messages: list[Any]) -> tuple[dict[str, int], bool]:
    prompt = completion = total = 0
    available = False
    for message in messages:
        metadata = getattr(message, "usage_metadata", None)
        if metadata is None and isinstance(message, dict):
            metadata = message.get("usage_metadata") or message.get("response_metadata", {}).get("token_usage")
        if not metadata:
            continue
        available = True
        prompt += int(metadata.get("input_tokens", metadata.get("prompt_tokens", 0)) or 0)
        completion += int(metadata.get("output_tokens", metadata.get("completion_tokens", 0)) or 0)
        total += int(metadata.get("total_tokens", 0) or 0)
    if available and not total:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}, available
