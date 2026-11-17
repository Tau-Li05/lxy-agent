from __future__ import annotations

import unittest
from unittest.mock import patch

from Agent.runtime.qwen_harness_v12 import _execute_math_v12, _math_tool_schemas_v12
from Agent.tools.math_tools import MathToolSession, execute_math_tool


class SmallCaseCrossCheckTests(unittest.TestCase):
    def _call(self, formula_code: str) -> dict:
        return {"name": "python_calculate", "args": {
            "code": "print(6)", "calculation_kind": "formula_count",
            "small_case": "Choose 2 of 4 objects",
            "small_case_bruteforce_code": (
                "from itertools import combinations\n"
                "print(len(list(combinations(range(4), 2))))"
            ),
            "small_case_formula_code": formula_code,
        }}

    def test_formula_count_schema_requires_kind(self):
        schemas = _math_tool_schemas_v12("python_exact", candidate_protocol=False)
        self.assertIn("calculation_kind", schemas[0]["parameters"]["required"])
        self.assertEqual(schemas[1]["name"], "exact_calculate")
        self.assertNotIn("calculation_kind", schemas[1]["parameters"]["properties"])

    def test_matching_small_case_allows_main_calculation(self):
        with patch("Agent.runtime.qwen_harness_v12.execute_math_tool", wraps=execute_math_tool) as run:
            result, check = _execute_math_v12(
                self._call("from math import comb\nprint(comb(4, 2))"), MathToolSession(), 2
            )
        self.assertTrue(result.success)
        self.assertEqual(check["status"], "passed")
        self.assertEqual((check["bruteforce_value"], check["formula_value"]), ("6", "6"))
        self.assertEqual(run.call_count, 2)

    def test_mismatch_blocks_main_calculation(self):
        with patch("Agent.runtime.qwen_harness_v12.execute_math_tool", wraps=execute_math_tool) as run:
            result, check = _execute_math_v12(self._call("print(5)"), MathToolSession(), 2)
        self.assertFalse(result.success)
        self.assertEqual(result.error["type"], "cross_check_mismatch")
        self.assertEqual(check["status"], "mismatch")
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
