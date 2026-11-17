from __future__ import annotations

import unittest

from Agent.runtime.math_workflow import MathEvidence
from Agent.runtime.qwen_harness_v14 import _new_alerts_v14
from Agent.runtime.qwen_harness_v15 import _final_evidence_v15, _update_evidence_versions_v15
from Agent.runtime.qwen_harness_v16 import _checkpoint_v16
from Agent.tools.math_tools import execute_math_tool


class BudgetMemoryVariantTests(unittest.TestCase):
    def test_v14_alert_is_generated_once_for_failure(self):
        call = {"id": "bad", "name": "python_calculate", "args": {"code": "print(1/0)"}}
        result = execute_math_tool(call["name"], call["args"])
        seen: set[str] = set()
        first = _new_alerts_v14(MathEvidence(), call, result, seen)
        second = _new_alerts_v14(MathEvidence(), call, result, seen)
        self.assertEqual(first[0]["kind"], "tool_failure")
        self.assertEqual(second, [])

    def test_v15_superseded_failure_is_absent_from_final_context(self):
        failed_call = {"id": "bad", "name": "python_calculate",
                       "args": {"code": "print(1/0)"}}
        fixed_call = {"id": "fixed", "name": "python_calculate",
                      "args": {"code": "print(3)"}}
        failure = execute_math_tool(failed_call["name"], failed_call["args"])
        success = execute_math_tool(fixed_call["name"], fixed_call["args"])
        evidence = MathEvidence()
        evidence.observe(failed_call, failure)
        evidence.observe(fixed_call, success)
        versions: list[dict] = []
        _update_evidence_versions_v15(versions, failed_call, failure)
        _update_evidence_versions_v15(versions, fixed_call, success,
                                      revision_claim=True)
        history = [
            {"role": "assistant", "content": "", "tool_calls": [failed_call]},
            {"role": "tool", "tool_call_id": "bad", "content": failure.output},
            {"role": "assistant", "content": "", "tool_calls": [fixed_call]},
            {"role": "tool", "tool_call_id": "fixed", "content": success.output},
        ]
        view, filtered, note = _final_evidence_v15(evidence, history, versions)
        self.assertEqual(note["invalidated_call_ids"], ["bad"])
        self.assertTrue(all(row["call_id"] != "bad" for row in view.results))
        self.assertNotIn("print(1/0)", str(filtered))
        self.assertIn("print(3)", str(filtered))

    def test_v16_checkpoint_does_not_promote_unchecked_candidate(self):
        evidence = MathEvidence()
        evidence.observe_model("#### 42")
        checkpoint = _checkpoint_v16(evidence, {"unverified_points": ["Need to verify the range"]})
        self.assertIsNone(checkpoint["candidate"])
        self.assertIn("verify the range", checkpoint["next_step"])


if __name__ == "__main__":
    unittest.main()
