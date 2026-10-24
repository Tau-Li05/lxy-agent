from __future__ import annotations

from typing import Any

from Agent.evaluation.models import AgentOutput
from Agent.runtime.direct_runtime import DirectLangChainRuntime


class BFCLProtocolRuntime(DirectLangChainRuntime):
    """Minimal BFCL-required loop using the shared protocol executor."""

    runtime_kind = "bfcl_protocol"

    async def run(self, item: dict[str, Any]) -> AgentOutput:
        output = await super().run(item)
        output.metadata["runtime"] = self.runtime_kind
        output.metadata["provider"] = self.profile.get("provider", "openai_compatible")
        output.metadata["exact_model_name"] = self.profile["name"]
        output.metadata["enhancements"] = {
            "planning": False, "verification": False, "compaction": False,
            "self_correction": False,
        }
        return output
