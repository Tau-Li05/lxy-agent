from __future__ import annotations

from typing import Any, Protocol

from .models import AgentOutput


class AgentRuntime(Protocol):
    async def run(self, agent_input: dict[str, Any]) -> AgentOutput: ...
