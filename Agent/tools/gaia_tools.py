from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Agent.tools.math_tools import MathToolResult, execute_math_tool


GAIA_TOOLS = [
    {
        "name": "search_engine_query", "description": "Search the web for current or factual information.",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "string"}, "max_results": {"type": "integer", "default": 10},
            "region": {"type": "string", "default": "wt-wt"}}, "required": ["keywords"]},
    },
    {
        "name": "fetch_url_content", "description": "Fetch and extract content from an HTTP or HTTPS URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "mode": {"type": "string", "enum": ["raw", "markdown", "truncate"]}},
            "required": ["url"]},
    },
    {
        "name": "read_attachment", "description": "Read the attachment supplied with the current GAIA task.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "python_calculate", "description": "Run short Python code for calculation or data analysis.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
]


def execute_gaia_tool(name: str, args: dict[str, Any], environment: dict[str, Any]) -> MathToolResult:
    if name == "python_calculate":
        return execute_math_tool(name, args, timeout_seconds=10)
    if name in {"search_engine_query", "fetch_url_content"}:
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
        except ImportError:
            import sys
            root = Path(__file__).resolve().parents[2] / "benchmark/BFCL-main/berkeley-function-call-leaderboard"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
        backend = WebSearchAPI()
        try:
            output = getattr(backend, name)(**args)
            return MathToolResult(json.dumps(output, ensure_ascii=False), not isinstance(output, dict) or "error" not in output)
        except Exception as exc:
            return MathToolResult(json.dumps({"error": type(exc).__name__, "message": str(exc)}), False,
                                  {"type": type(exc).__name__})
    if name == "read_attachment":
        raw_path = environment.get("attachment")
        if not raw_path:
            return MathToolResult(json.dumps({"error": "this task has no attachment"}), False)
        path = Path(raw_path).resolve()
        allowed = (Path(__file__).resolve().parents[2] / "benchmark/GAIA").resolve()
        if allowed not in path.parents:
            return MathToolResult(json.dumps({"error": "attachment path is outside GAIA"}), False)
        try:
            suffix = path.suffix.lower()
            if suffix in {".txt", ".csv", ".json", ".jsonld", ".xml", ".py"}:
                content = path.read_text(encoding="utf-8", errors="replace")
            elif suffix == ".pdf":
                from pypdf import PdfReader
                content = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            elif suffix == ".xlsx":
                import pandas as pd
                sheets = pd.read_excel(path, sheet_name=None)
                content = "\n\n".join(f"[{name}]\n{frame.to_csv(index=False)}" for name, frame in sheets.items())
            else:
                return MathToolResult(json.dumps({"error": f"unsupported attachment type: {suffix}"}), False)
            return MathToolResult(json.dumps({"content": content[-100000:]}, ensure_ascii=False), True)
        except Exception as exc:
            return MathToolResult(json.dumps({"error": type(exc).__name__, "message": str(exc)}), False,
                                  {"type": type(exc).__name__})
    return MathToolResult(json.dumps({"error": f"unknown GAIA tool: {name}"}), False)
