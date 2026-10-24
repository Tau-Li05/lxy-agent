from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


class SystemPromptBuilder:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def build(self, base: str, config: dict[str, Any], *, tool_names: list[str] | None = None) -> str:
        policy = config.get("features", {}).get("system_prompt", {})
        if policy and not policy.get("enabled", True):
            return ""
        sections: list[tuple[str, str]] = []
        if policy.get("base", True) and base.strip():
            sections.append(("base_instructions", base.strip()))
        if policy.get("project_rules", False):
            path = self.project_root / "AGENTS.md"
            if path.exists():
                sections.append(("project_rules", path.read_text(encoding="utf-8").strip()))
        if policy.get("environment_info", False):
            environment = [f"Python: {platform.python_version()}"]
            if tool_names:
                environment.append("Available tools: " + ", ".join(tool_names))
            sections.append(("environment", "\n".join(environment)))
        return "\n\n".join(f"<{name}>\n{text}\n</{name}>" for name, text in sections)
