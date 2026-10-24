from __future__ import annotations

import json
import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


PYTHON_TOOL = {
    "name": "python_calculate",
    "description": "Run short deterministic Python code for exact mathematical calculation or verification. No network or file access is intended. Print the relevant result.",
    "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}},
        "required": ["code"], "additionalProperties": False,
    },
}

SYMPY_TOOL = {
    "name": "sympy_calculate",
    "description": "Run a short SymPy calculation to simplify, factor, solve, or verify a mathematical expression. Print the relevant result.",
    "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}},
        "required": ["code"], "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class MathToolResult:
    output: str
    success: bool
    error: dict[str, Any] | None = None


def math_tool_schemas(mode: str) -> list[dict[str, Any]]:
    if mode == "python":
        return [PYTHON_TOOL]
    if mode == "python_sympy":
        return [PYTHON_TOOL, SYMPY_TOOL]
    return []


def execute_math_tool(name: str, args: dict[str, Any], *, timeout_seconds: float = 5.0) -> MathToolResult:
    if name not in {"python_calculate", "sympy_calculate"}:
        return MathToolResult(json.dumps({"error": f"unknown math tool: {name}"}), False,
                              {"type": "unknown_tool"})
    code = str(args.get("code", ""))
    if len(code) > 10000:
        return MathToolResult(json.dumps({"error": "code exceeds 10000 characters"}), False,
                              {"type": "input_too_large"})
    violation = _code_violation(code)
    if violation:
        return MathToolResult(json.dumps({"error": violation}), False, {"type": "unsafe_code"})
    # Isolated interpreter, bounded time and output. This is a computation aid,
    # not a security boundary against hostile code.
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return MathToolResult(json.dumps({"error": "tool timeout"}), False, {"type": "timeout"})
    stdout, stderr = completed.stdout[-50000:], completed.stderr[-10000:]
    payload = {"stdout": stdout, "stderr": stderr, "exit_code": completed.returncode}
    return MathToolResult(json.dumps(payload, ensure_ascii=False), completed.returncode == 0,
                          None if completed.returncode == 0 else {"type": "nonzero_exit"})


def _code_violation(code: str) -> str | None:
    allowed_imports = {"math", "fractions", "decimal", "itertools", "functools", "collections", "sympy"}
    banned_calls = {"open", "exec", "eval", "compile", "input", "__import__", "breakpoint"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"invalid Python syntax: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in allowed_imports for alias in node.names):
                return "only deterministic math-library imports are allowed"
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".", 1)[0] not in allowed_imports:
                return "only deterministic math-library imports are allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
            return f"call to {node.func.id} is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
    return None
