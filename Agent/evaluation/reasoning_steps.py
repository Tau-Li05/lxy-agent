from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def extract_visible_steps(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for block in blocks:
        try:
            value = json.loads(block)
            if isinstance(value.get("steps"), list):
                return [step if isinstance(step, dict) else {"claim": str(step)} for step in value["steps"]]
        except (ValueError, TypeError):
            pass
    matches = list(re.finditer(r"(?im)^\s*(?:\*\*)?(?:step|步骤)\s*(\d+)\s*[:：.-]?\s*(.*)$", text))
    steps: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = (match.group(2) + "\n" + text[match.end():end]).strip()
        steps.append({"step_id": f"s{match.group(1)}", "claim": body, "depends_on": []})
    return steps


def evaluate_steps(text: str) -> dict[str, Any]:
    steps = extract_visible_steps(text)
    evaluations = []
    for step in steps:
        claim = " ".join(str(step.get(key, "")) for key in ("claim", "derivation", "result"))
        verdict = _evaluate_numeric_equality(claim)
        evaluations.append({"step_id": step.get("step_id"), "verdict": verdict,
                            "method": "safe_numeric_equality" if verdict != "not_evaluable" else None})
    evaluated = [row for row in evaluations if row["verdict"] != "not_evaluable"]
    correct = sum(row["verdict"] == "correct" for row in evaluated)
    return {"steps": steps, "evaluations": evaluations, "total_steps": len(steps),
            "evaluated_steps": len(evaluated), "correct_steps": correct,
            "step_accuracy": correct / len(evaluated) if evaluated else None,
            "evaluated_step_coverage": len(evaluated) / len(steps) if steps else 0.0}


def aggregate_step_accuracy(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    details = [evaluate_steps(str(output.get("final_answer") or "")) for output in outputs]
    total = sum(row["total_steps"] for row in details)
    evaluated = sum(row["evaluated_steps"] for row in details)
    correct = sum(row["correct_steps"] for row in details)
    return {"correct_steps": correct, "evaluated_steps": evaluated, "total_steps": total,
            "step_accuracy": correct / evaluated if evaluated else None,
            "evaluated_step_coverage": evaluated / total if total else 0.0,
            "method": "safe_numeric_equality_v1"}


def _evaluate_numeric_equality(text: str) -> str:
    candidates = []
    for line in text.splitlines():
        if line.count("=") != 1:
            continue
        left, right = (part.strip() for part in line.split("=", 1))
        left = re.sub(r"^[^0-9+().-]*", "", left)
        right = re.sub(r"[^0-9+*/(). ^-].*$", "", right).strip()
        if left and right and re.fullmatch(r"[0-9+*/(). ^-]+", left) and re.fullmatch(r"[0-9+*/(). ^-]+", right):
            candidates.append((left, right))
    if not candidates:
        return "not_evaluable"
    left, right = candidates[-1]
    try:
        return "correct" if abs(float(_safe_eval(left)) - float(_safe_eval(right))) < 1e-9 else "incorrect"
    except Exception:
        return "not_evaluable"


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression.replace("^", "**").strip(), mode="eval").body
    def visit(value: ast.AST) -> float:
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return float(value.value)
        if isinstance(value, ast.BinOp) and type(value.op) in _OPS:
            return _OPS[type(value.op)](visit(value.left), visit(value.right))
        if isinstance(value, ast.UnaryOp) and type(value.op) in _OPS:
            return _OPS[type(value.op)](visit(value.operand))
        raise ValueError("unsupported expression")
    return visit(node)
