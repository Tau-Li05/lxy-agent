from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


ArtifactWriter = Callable[[dict[str, Any]], Any]


def estimate_tokens(messages: list[Any]) -> int:
    """Tokenizer-independent estimate used only for context budget decisions."""
    text = json.dumps(messages, ensure_ascii=False, default=str)
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass(frozen=True)
class ContextPolicy:
    enabled: bool = False
    trigger_ratio: float = 0.80
    target_ratio: float = 0.65
    tool_result_max_characters: int = 50_000
    tool_result_preview_characters: int = 2_000
    preserve_recent_messages: int = 8
    collapse_summary_characters: int = 8_000


@dataclass
class ContextResult:
    messages: list[Any]
    original_tokens: int
    compacted_tokens: int
    original_characters: int
    compacted_characters: int
    actions: list[dict[str, Any]]

    @property
    def changed(self) -> bool:
        return bool(self.actions)


class ContextManager:
    """Deterministic five-level context reduction with auditable actions."""

    def __init__(self, policy: ContextPolicy, *, context_window: int, reserved_output_tokens: int) -> None:
        self.policy = policy
        self.input_budget = max(1, context_window - reserved_output_tokens)
        self.trigger_tokens = max(1, int(self.input_budget * policy.trigger_ratio))
        self.target_tokens = max(1, int(self.input_budget * policy.target_ratio))

    @classmethod
    def from_config(cls, feature: dict[str, Any], profile: dict[str, Any],
                    context_window_override: int | None = None) -> "ContextManager":
        policy = ContextPolicy(
            enabled=bool(feature.get("enabled", False)),
            trigger_ratio=float(feature.get("trigger_ratio", 0.80)),
            target_ratio=float(feature.get("target_ratio", 0.65)),
            tool_result_max_characters=int(feature.get("tool_result_max_characters", 50_000)),
            tool_result_preview_characters=int(feature.get("tool_result_preview_characters", 2_000)),
            preserve_recent_messages=int(feature.get("preserve_recent_messages", 8)),
            collapse_summary_characters=int(feature.get("collapse_summary_characters", 8_000)),
        )
        return cls(policy, context_window=int(context_window_override or profile["context_window"]),
                   reserved_output_tokens=int(profile["max_tokens"]))

    def needs_compaction(self, messages: list[Any]) -> bool:
        if not self.policy.enabled:
            return False
        if estimate_tokens(messages) >= self.trigger_tokens:
            return True
        return any(
            isinstance(message, dict) and message.get("role") == "tool"
            and len(str(message.get("content", ""))) > self.policy.tool_result_max_characters
            for message in messages
        )

    def compact(self, messages: list[Any], *, artifact_writer: ArtifactWriter | None = None) -> ContextResult:
        original = deepcopy(messages)
        managed = deepcopy(messages)
        actions: list[dict[str, Any]] = []
        removed: list[Any] = []

        # Level 1: preserve oversized tool results outside the prompt and retain a reachable reference.
        for index, message in enumerate(managed):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = str(message.get("content", ""))
            if len(content) <= self.policy.tool_result_max_characters:
                continue
            artifact = artifact_writer({"kind": "context_tool_result", "content": content}) if artifact_writer else None
            message["content"] = json.dumps({
                "context_notice": "Oversized tool result externalized",
                "artifact": artifact,
                "preview": content[:self.policy.tool_result_preview_characters],
                "original_characters": len(content),
            }, ensure_ascii=False)
            actions.append({"level": "tool_result_budget", "message_index": index,
                            "original_characters": len(content)})

        # Level 2: remove the oldest complete history region while retaining system and recent context.
        if estimate_tokens(managed) > self.target_tokens:
            system_count = 1 if managed and isinstance(managed[0], dict) and managed[0].get("role") == "system" else 0
            keep_from = max(system_count, len(managed) - self.policy.preserve_recent_messages)
            while keep_from > system_count and isinstance(managed[keep_from], dict) and managed[keep_from].get("role") == "tool":
                keep_from -= 1
            if keep_from > system_count:
                removed = managed[system_count:keep_from]
                managed = [*managed[:system_count], *managed[keep_from:]]
                actions.append({"level": "history_snip", "removed_messages": len(removed)})

        # Level 3: compact older retained tool results without touching the newest observations.
        if estimate_tokens(managed) > self.target_tokens:
            recent_boundary = max(0, len(managed) - 2)
            for index, message in enumerate(managed[:recent_boundary]):
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                content = str(message.get("content", ""))
                if len(content) <= self.policy.tool_result_preview_characters:
                    continue
                message["content"] = content[:self.policy.tool_result_preview_characters] + "\n[older tool result compacted]"
                actions.append({"level": "microcompact", "message_index": index,
                                "original_characters": len(content)})

        # Level 4: retain a bounded, deterministic trace of history removed at level 2.
        if removed:
            summary_lines = []
            for message in removed:
                if isinstance(message, dict):
                    role, content = message.get("role", "unknown"), str(message.get("content", ""))
                else:
                    role, content = "unknown", str(message)
                summary_lines.append(f"{role}: {content[:500]}")
            summary = "[Collapsed earlier context]\n" + "\n".join(summary_lines)
            summary = summary[:self.policy.collapse_summary_characters]
            insert_at = 1 if managed and isinstance(managed[0], dict) and managed[0].get("role") == "system" else 0
            managed.insert(insert_at, {"role": "user", "content": summary})
            actions.append({"level": "context_collapse", "summary_characters": len(summary)})

        # Level 5: deterministic emergency reduction if the collapse still exceeds the target.
        if estimate_tokens(managed) > self.target_tokens:
            protected = {0, len(managed) - 1}
            for index, message in enumerate(managed):
                if estimate_tokens(managed) <= self.target_tokens:
                    break
                if index in protected or not isinstance(message, dict):
                    continue
                content = str(message.get("content", ""))
                if len(content) <= 500:
                    continue
                message["content"] = content[:500] + "\n[autocompact emergency truncation]"
                actions.append({"level": "autocompact", "message_index": index,
                                "original_characters": len(content)})

        def characters(value: list[Any]) -> int:
            return len(json.dumps(value, ensure_ascii=False, default=str))

        return ContextResult(
            messages=managed,
            original_tokens=estimate_tokens(original), compacted_tokens=estimate_tokens(managed),
            original_characters=characters(original), compacted_characters=characters(managed), actions=actions,
        )
