from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


PYTHON_TOOL = {
    "name": "python_calculate",
    "description": (
        "Run short deterministic Python for enumeration, exact calculation or verification. "
        "You may import math, fractions, decimal, itertools, functools, collections and sympy. "
        "Use Fraction or SymPy Rational for exact fractions. Preserve the original problem constraints; "
        "print only the relevant result. For a final candidate, print a JSON object with candidate_answer "
        "and method; bare numbers are treated as intermediate results. For a finite search, optionally "
        "print a JSON object with checked_count, search_bounds, and search_complete; these are reports, "
        "not proof that the range covers the original question. For purpose=enumerate, provide "
        "constraints and execute assertions checking them. No network or file access. "
        "On failure, fix the code before retrying."
    ),
    "parameters": {
        "type": "object", "properties": {
            "code": {"type": "string"},
            "purpose": {"type": "string", "enum": ["calculate", "enumerate", "verify"]},
            "constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
        },
        "required": ["code"], "additionalProperties": False,
    },
}

EXACT_TOOL = {
    "name": "exact_calculate",
    "description": (
        "Evaluate a short exact integer/rational expression, e.g. 9+18+252 or 1/3+0.1. "
        "Supports + - * / // % **, parentheses and abs, gcd, lcm, factorial, comb, isqrt. "
        "Use ** for powers. No variables, strings, assignments or Python code. "
        "Returns an exact integer or reduced fraction; use Python for enumeration or symbolic algebra."
    ),
    "parameters": {
        "type": "object", "properties": {"expression": {"type": "string"}},
        "required": ["expression"], "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class MathToolResult:
    output: str
    success: bool
    error: dict[str, Any] | None = None


def _result(*, success: bool, error_type: str | None = None, message: str | None = None,
            line: int | None = None, stdout: str = "", stderr: str = "",
            exit_code: int | None = None, truncated: bool = False, **extra: Any) -> MathToolResult:
    error = {"type": error_type, "message": message, "line": line} if error_type else None
    payload = {"success": success, "error": error, "error_type": error_type,
               "message": message, "line": line, "stdout": stdout, "stderr": stderr,
               "exit_code": exit_code, "timed_out": error_type == "timeout",
               "truncated": truncated, **extra}
    return MathToolResult(json.dumps(payload, ensure_ascii=False), success, error)


def math_tool_schemas(mode: str, *, candidate_protocol: bool = False) -> list[dict[str, Any]]:
    # Keep the v2 schema/prompt fixed unless the protocol is explicitly being tested.
    if mode not in {"python", "python_sympy", "python_exact"}:
        return []
    python_tool = deepcopy(PYTHON_TOOL)
    if not candidate_protocol:
        python_tool["description"] = (
            "Run short deterministic Python for enumeration, exact calculation or verification. "
            "You may import math, fractions, decimal, itertools, functools, collections and sympy. "
            "Use Fraction or SymPy Rational for exact fractions. Preserve the original problem constraints; "
            "print only the relevant result. No network or file access. On failure, fix the code before retrying."
        )
        python_tool["parameters"]["properties"] = {"code": {"type": "string"}}
    return [python_tool, deepcopy(EXACT_TOOL)]


class _ExpressionLimit(ValueError):
    pass


def _exact_value(expression: str) -> Fraction:
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise _ExpressionLimit("expression exceeds 128 syntax nodes")

    def bounded(value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise _ExpressionLimit("exact result exceeds 4096 bits")
        return value

    def integer(value: Fraction) -> int:
        if value.denominator != 1:
            raise ValueError("this operation requires integer arguments")
        return value.numerator

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            literal = ast.get_source_segment(expression, node) or ""
            if len(literal) > 128:
                raise _ExpressionLimit("numeric literal exceeds 128 characters")
            if type(node.value) is float and not math.isfinite(node.value):
                raise _ExpressionLimit("numeric literal is too large")
            exponent = re.search(r"[eE]([+-]?[0-9_]+)$", literal)
            if type(node.value) is float and exponent and abs(int(exponent[1].replace('_', ''))) > 1000:
                raise _ExpressionLimit("decimal exponent exceeds 1000")
            # Fraction parses decimal text exactly, avoiding binary float rounding.
            return bounded(Fraction(node.value if type(node.value) is int else literal.replace('_', '')))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.FloorDiv): value = Fraction(left // right)
            elif isinstance(node.op, ast.Mod): value = left % right
            elif isinstance(node.op, ast.Pow):
                exponent = integer(right)
                if abs(exponent) > 4096:
                    raise _ExpressionLimit("power exponent exceeds 4096")
                if max(left.numerator.bit_length(), left.denominator.bit_length()) * abs(exponent) > 8192:
                    raise _ExpressionLimit("power would exceed result size limit")
                value = left ** exponent
            else: raise ValueError("unsupported arithmetic operator; use ** for powers")
            return bounded(Fraction(value))
        functions = {"abs": (1, abs), "gcd": (2, math.gcd), "lcm": (2, math.lcm),
                     "factorial": (1, math.factorial), "comb": (2, math.comb), "isqrt": (1, math.isqrt)}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            name = node.func.id
            arity, function = functions[name]
            if node.keywords or len(node.args) != arity:
                raise ValueError(f"{name} requires {arity} positional arguments")
            values = [visit(arg) for arg in node.args]
            if name == "abs": return abs(values[0])
            args = [integer(value) for value in values]
            if name in {"factorial", "comb"} and any(arg < 0 or arg > 1000 for arg in args):
                raise _ExpressionLimit("factorial/comb arguments must be between 0 and 1000")
            return bounded(Fraction(function(*args)))
        raise ValueError("only numeric literals, arithmetic and the documented functions are allowed")

    return visit(tree.body)


# Limit output while it is generated, rather than collecting an unbounded stdout first.
# This is a computation aid, not a security boundary against hostile Python code.
_PYTHON_RUNNER = r'''
import sys, json, traceback
class BoundedOutput:
    def __init__(self, limit):
        self.limit, self.parts, self.size, self.truncated = limit, [], 0, False
    def write(self, text):
        remaining = max(0, self.limit-self.size)
        self.parts.append(text[:remaining]) if remaining else None
        self.size += min(len(text), remaining)
        self.truncated |= len(text) > remaining
        return len(text)
    def flush(self): pass
out, err = BoundedOutput(4000), BoundedOutput(2000)
original = sys.stdout
sys.stdout, sys.stderr = out, err
error_type = message = line = None
exit_code = 0
assertions_passed = [0]
def mark_assertion():
    assertions_passed[0] += 1
try:
    import ast
    class TrackAssertions(ast.NodeTransformer):
        def visit_Assert(self, node):
            return [node, ast.copy_location(ast.Expr(ast.Call(
                func=ast.Name(id='_harness_assert_passed', ctx=ast.Load()), args=[], keywords=[])), node)]
    tree = ast.fix_missing_locations(TrackAssertions().visit(ast.parse(sys.argv[1])))
    exec(compile(tree, '<math_tool>', 'exec'), {
        '__name__': '__main__', '_harness_assert_passed': mark_assertion})
except BaseException as exc:
    error_type, message, exit_code = type(exc).__name__, str(exc)[:1000], 1
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == '<math_tool>']
    line = frames[-1].lineno if frames else None
    traceback.print_exc(limit=4)
finally:
    sys.stdout = original
print(json.dumps({'success': exit_code == 0, 'error_type': error_type, 'message': message,
    'line': line, 'stdout': ''.join(out.parts), 'stderr': ''.join(err.parts),
    'exit_code': exit_code, 'assertions_passed': assertions_passed[0], 'truncated': out.truncated or err.truncated}, ensure_ascii=False))
'''


def execute_math_tool(name: str, args: dict[str, Any], *, timeout_seconds: float = 5.0) -> MathToolResult:
    # Accept the old tool name for saved/replayed requests, but do not advertise it.
    if name not in {"python_calculate", "sympy_calculate", "exact_calculate"}:
        return _result(success=False, error_type="unknown_tool", message=f"unknown math tool: {name}")
    key = "expression" if name == "exact_calculate" else "code"
    allowed = {key} if name == "exact_calculate" else {key, "purpose", "constraints"}
    if not isinstance(args, dict) or set(args) - allowed or not isinstance(args.get(key), str) or not args[key].strip():
        return _result(success=False, error_type="invalid_arguments", message=f"provide one nonempty string: {key}")
    purpose = args.get("purpose", "calculate")
    constraints = args.get("constraints", [])
    if (purpose not in {"calculate", "enumerate", "verify"} or not isinstance(constraints, list)
            or len(constraints) > 16 or any(not isinstance(c, str) or not c.strip() for c in constraints)):
        return _result(success=False, error_type="invalid_arguments", message="invalid purpose or constraints")
    if purpose == "enumerate" and not constraints:
        return _result(success=False, error_type="constraint_check_required", message="enumeration requires original constraints and executed assertions")
    source = args[key].strip()
    if len(source) > (2000 if name == "exact_calculate" else 10000):
        return _result(success=False, error_type="input_too_large", message=f"{key} exceeds length limit")
    if name == "exact_calculate":
        try:
            value = _exact_value(source)
            return _result(success=True, result=str(value), numerator=str(value.numerator),
                           denominator=str(value.denominator), stdout=str(value), exit_code=0)
        except (ArithmeticError, ValueError, SyntaxError, RecursionError) as exc:
            return _result(success=False, error_type=type(exc).__name__ if not isinstance(exc, _ExpressionLimit) else "expression_limit",
                           message=str(exc)[:1000], line=getattr(exc, "lineno", None))
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return _result(success=False, error_type=type(exc).__name__, message=str(exc)[:1000], line=getattr(exc, "lineno", None))
    violation = _code_violation(source)
    if violation:
        return _result(success=False, error_type="unsafe_code", message=violation)
    try:
        completed = subprocess.run([sys.executable, "-I", "-c", _PYTHON_RUNNER, source],
                                   capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return _result(
            success=False, error_type="timeout",
            message=(f"calculation exceeded {timeout_seconds:g} seconds. Derive finite search bounds from the "
                     "original constraints, reduce the search space or change the algorithm, then print "
                     "the number of candidates checked and only the matching results. Do not rerun unchanged code."),
        )
    try:
        payload = json.loads(completed.stdout)
    except (ValueError, TypeError):
        return _result(success=False, error_type="execution_error", message="interpreter did not return a structured result",
                       stderr=completed.stderr[-2000:], exit_code=completed.returncode, truncated=len(completed.stderr)>2000)
    if payload.get("success") and purpose == "enumerate" and not payload.get("assertions_passed"):
        payload.update(success=False, error_type="constraint_check_required",
                       message="enumeration must execute assertions for the stated constraints")
    if payload.get("success") and payload.get("truncated"):
        payload.update(success=False, error_type="output_truncated",
                       message="Python ran, but its printed output was truncated, so the result is not usable. Rerun with concise output: print the candidate count, only matches that pass all original constraints, and the final value; remove per-item debug prints.",
                       execution_succeeded=True, usable_result=False)
    elif payload.get("success") and not str(payload.get("stdout", "")).strip():
        payload.update(success=False, error_type="empty_output",
                       message="Python executed successfully but produced no usable stdout. Print the result (for example, print(roots)); change the code before retrying.",
                       execution_succeeded=True, usable_result=False)
    elif payload.get("success"):
        payload.update(execution_succeeded=True, usable_result=True)
    return _result(**payload)


def _code_violation(code: str) -> str | None:
    allowed_imports = {"math", "fractions", "decimal", "itertools", "functools", "collections", "sympy", "json"}
    banned_calls = {"open", "exec", "eval", "compile", "input", "__import__", "breakpoint"}
    tree = ast.parse(code)
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


class MathToolSession:
    """Per-attempt failure cache, including concurrent duplicates in a tool batch."""

    def __init__(self, *, cache_successes: bool = False, repair_turn_enabled: bool = False) -> None:
        self.cache_successes = cache_successes
        self.repair_turn_enabled = repair_turn_enabled
        self._successes: dict[str, MathToolResult] = {}
        self._failures: dict[str, MathToolResult] = {}
        self._blocked_repeats: dict[str, int] = {}
        self._last_error_type: dict[str, str] = {}
        self._same_error_streak: dict[str, int] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def execute(self, name: str, args: dict[str, Any], *, timeout_seconds: float = 5.0) -> MathToolResult:
        canonical_name = "python_calculate" if name == "sympy_calculate" else name
        key = json.dumps([canonical_name, args], sort_keys=True, ensure_ascii=False)
        if isinstance(args, dict):
            field = "expression" if name == "exact_calculate" else "code"
            if isinstance(args.get(field), str):
                try:
                    tree = ast.parse(args[field].strip(), mode="eval" if field == "expression" else "exec")
                    # Include additional args: invalid calls must not poison valid ones.
                    key = json.dumps([canonical_name, ast.dump(tree), {k:v for k,v in args.items() if k != field}], sort_keys=True)
                except (SyntaxError, ValueError, RecursionError):
                    pass
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if self.cache_successes and key in self._successes:
                previous = self._successes[key]
                payload = json.loads(previous.output)
                payload["cached_success"] = True
                return MathToolResult(json.dumps(payload, ensure_ascii=False), True)
            if key in self._failures:
                previous = self._failures[key]
                repeats = self._blocked_repeats.get(key, 0) + 1
                self._blocked_repeats[key] = repeats
                repair_required = self.repair_turn_enabled and repeats == 2
                return _result(success=False, error_type="repair_required" if repair_required else "duplicate_failed_call",
                               message=("This is the second blocked repeat of the same failed call. Pause tool use and explain the original error plus a specific change to the code or search method."
                                        if repair_required else
                                        "This same tool call already produced a failed or unusable result. Change the code/arguments or use another method; do not repeat it."),
                               previous_error=previous.error, previous_result=json.loads(previous.output),
                               blocked_repeat_count=repeats, deduplicated=True)
            result = execute_math_tool(name, args, timeout_seconds=timeout_seconds)
            if not result.success:
                self._failures[key] = result
            elif self.cache_successes:
                self._successes[key] = result
            error_type = result.error.get("type") if result.error else None
            with self._guard:
                if error_type in {"TypeError", "AttributeError"}:
                    streak = (self._same_error_streak.get(canonical_name, 0) + 1
                              if self._last_error_type.get(canonical_name) == error_type else 1)
                    self._last_error_type[canonical_name] = error_type
                    self._same_error_streak[canonical_name] = streak
                else:
                    self._last_error_type.pop(canonical_name, None)
                    self._same_error_streak.pop(canonical_name, None)
                    streak = 0
            if streak >= 2 and isinstance(args, dict) and isinstance(args.get("code"), str):
                payload = json.loads(result.output)
                line = result.error.get("line") if result.error else None
                lines = args["code"].splitlines()
                failing_line = lines[line - 1].strip() if isinstance(line, int) and 1 <= line <= len(lines) else ""
                indexed = []
                if failing_line:
                    try:
                        indexed = list(dict.fromkeys(node.value.id for node in ast.walk(ast.parse(failing_line))
                                                     if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)))
                    except (SyntaxError, ValueError):
                        pass
                probes = [f"print(type({value}), repr({value}))" for value in indexed[:3]]
                advice = ("The same error type occurred in consecutive changed calculations. "
                          "Before rerunning the full program, make a small diagnostic call with the required setup "
                          "and print the types and values used on the failing line.")
                payload["same_error_diagnostic"] = {
                    "error_type": error_type, "streak": streak, "failing_line": failing_line,
                    "suggested_probes": probes, "advice": advice,
                }
                payload["message"] = f"{payload.get('message') or error_type} {advice}"
                result = MathToolResult(json.dumps(payload, ensure_ascii=False), False, result.error)
            return result
