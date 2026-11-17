from __future__ import annotations

import unittest

from Agent.runtime.math_workflow import MathEvidence
from Agent.runtime.qwen_harness_v13 import _after_tool_v13, _memory_prompt_v13, _question_memory_v13
from Agent.tools.math_tools import execute_math_tool


class ConstraintEvidenceMemoryTests(unittest.TestCase):
    def test_original_condition_quotes_survive_tool_updates(self):
        question = [{"role": "user", "content": (
            "Count subsets of five chairs such that no person sits next to two other people. "
            "What is the number of valid subsets?"
        )}]
        memory = _question_memory_v13(question)
        original = list(memory["condition_quotes"])
        call = {"id": "a", "name": "python_calculate", "args": {
            "code": "print(2)"}}
        result = execute_math_tool("python_calculate", call["args"])
        evidence = MathEvidence()
        evidence.observe(call, result)
        _after_tool_v13(memory, call, result, evidence)
        self.assertEqual(memory["condition_quotes"], original)
        self.assertIn("two other people", memory["condition_quotes"][0])
        self.assertEqual(memory["tool_evidence"][-1]["condition_coverage"],
                         "not_mechanically_verified")
        self.assertIn("original conditions", _memory_prompt_v13(memory))

    def test_failed_result_is_not_saved_as_successful_evidence(self):
        memory = _question_memory_v13([{"role": "user", "content": "Find x if x is positive."}])
        call = {"id": "bad", "name": "python_calculate", "args": {"code": "print(1/0)"}}
        result = execute_math_tool("python_calculate", call["args"])
        _after_tool_v13(memory, call, result, MathEvidence())
        self.assertFalse(memory["tool_evidence"][-1]["success"])
        self.assertEqual(memory["tool_evidence"][-1]["stdout_excerpt"], "")
        self.assertEqual(memory["open_mismatches"][-1]["kind"], "tool_failure")


if __name__ == "__main__":
    unittest.main()
