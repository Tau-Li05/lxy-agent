from __future__ import annotations

import unittest
from unittest.mock import patch

from Agent.engine import estimate_tokens
from Agent.runtime.math_workflow import MathEvidence
from Agent.tests import test_math_tool_workflow as workflow_tests
from Agent.tests.test_math_tool_workflow import Model, tool_message
from Agent.tests.test_runtime_v3 import FakeMessage
from Agent.tools.math_tools import execute_math_tool, math_tool_schemas


def checked_ledger():
    ledger = MathEvidence()
    request = {"id": "candidate", "name": "python_calculate", "args": {
        "code": 'assert 390+3==393\nprint(\'{"candidate_answer":393}\')'}}
    ledger.observe(request, execute_math_tool(request["name"], request["args"]))
    return ledger


class FinalizationContextTests(unittest.TestCase):
    def test_empty_ledger_preserves_earlier_definition_beyond_old_character_tail(self):
        ledger = MathEvidence()
        question = [{"role": "user", "content": "Original geometry constraints."}]
        text = "Let t denote the original tangent parameter, t=7/16.\n\n" + "work\n\n" * 500
        messages = ledger.final_messages("Finish with #### integer.", question, text,
                                        history=[*question, FakeMessage(text)], max_input_tokens=10000)
        self.assertIn("t=7/16", str(messages))
        self.assertEqual(ledger.finalization_info["mode"], "continue_history")
        self.assertNotIn("Extract the supported candidate", str(messages))
        self.assertIn(question[0], messages)

    def test_oversized_reasoning_keeps_whole_definition_and_recent_conclusion(self):
        text = "Definition: x means the original side length.\n\n" + "Long algebraic development.\n\n" * 500
        text += "Latest conclusion: x=245; check the integer constraint."
        ledger = MathEvidence()
        messages = ledger.final_messages("Answer concisely.", [{"role": "user", "content": "Find x."}],
                                        text, history=[FakeMessage(text)], max_input_tokens=1800)
        self.assertIn("Definition: x means", str(messages))
        self.assertIn("Latest conclusion: x=245", str(messages))
        self.assertIn("omitted for input budget", str(messages))
        self.assertLessEqual(ledger.finalization_info["input_tokens_estimate"], 1800)

    def test_complete_tool_exchange_survives_as_data_without_orphan_provider_tools(self):
        ledger = MathEvidence()
        history = [tool_message("source", "print(279)"),
                   {"role": "tool", "tool_call_id": "source", "content": "279"}]
        messages = ledger.final_messages("Finish.", [{"role": "user", "content": "Question"}], None,
                                        history=history, max_input_tokens=4000)
        self.assertIn("print(279)", str(messages))
        self.assertIn('"tool_call_id": "source"', messages[-1]["content"])
        self.assertFalse(any(message.get("role") == "tool" for message in messages))
        self.assertEqual(ledger.finalization_info["retained_groups"], 1)

    def test_only_checked_nonconflicting_candidate_qualifies_for_extraction(self):
        ledger = checked_ledger()
        self.assertTrue(ledger.can_extract)
        messages = ledger.final_messages("Finish.", [], None)
        self.assertEqual(ledger.finalization_info["mode"], "extract")
        self.assertIn("Extract the supported candidate", str(messages))
        ledger.observe_model("#### 110")
        self.assertFalse(ledger.can_extract)
        ledger.final_messages("Finish.", [], None)
        self.assertEqual(ledger.finalization_info["mode"], "continue_history")

    def test_new_failure_invalidates_compact_extraction(self):
        ledger = checked_ledger()
        ledger.observe({"id": "bad", "name": "python_calculate", "args": {"code": "assert False"}},
                       execute_math_tool("python_calculate", {"code": "assert False"}))
        self.assertFalse(ledger.can_extract)

    def test_unchecked_or_failed_partial_answer_is_never_an_extraction_gate(self):
        for code in ['print(\'{"candidate_answer":393}\')',
                     'print(\'{"candidate_answer":393}\')\nassert False']:
            ledger = MathEvidence()
            request = {"id": "c", "name": "python_calculate", "args": {"code": code}}
            ledger.observe(request, execute_math_tool("python_calculate", request["args"]))
            self.assertFalse(ledger.can_extract)

    def test_history_strategy_is_independent_of_candidate_presence(self):
        ledger = checked_ledger()
        ledger.final_messages("Finish.", [], None, strategy="history")
        self.assertEqual(ledger.finalization_info["mode"], "continue_history")

    def test_default_tool_schema_matches_simple_code_protocol_without_mutation(self):
        basic = math_tool_schemas("python_exact")[0]
        extended = math_tool_schemas("python_exact", candidate_protocol=True)[0]
        self.assertEqual(set(basic["parameters"]["properties"]), {"code"})
        self.assertNotIn("candidate_answer", basic["description"])
        self.assertIn("purpose", extended["parameters"]["properties"])
        self.assertEqual(set(math_tool_schemas("python_exact")[0]["parameters"]["properties"]), {"code"})

    def test_message_estimate_includes_object_body(self):
        short = estimate_tokens([FakeMessage("short")])
        long = estimate_tokens([FakeMessage("reasoning " * 1000)])
        self.assertGreater(long - short, 900)


class FinalizationWorkflowTests(unittest.IsolatedAsyncioTestCase):
    config = workflow_tests.MathWorkflowTests.config
    run_case = workflow_tests.MathWorkflowTests.run_case

    async def test_evidence_is_passive_by_default_without_cache_or_early_stop(self):
        model = Model([tool_message("a", 'print(\'{"candidate_answer":393}\')'),
                       tool_message("b", 'print(\'{"candidate_answer":393}\')'),
                       tool_message("c", 'print(\'{"candidate_answer":393}\')'), FakeMessage("#### 393")])
        with patch("Agent.tools.math_tools.execute_math_tool", wraps=execute_math_tool) as execute:
            output, events = await self.run_case(self.config(), model)
        self.assertEqual(output.status, "success")
        self.assertEqual(execute.call_count, 3)
        self.assertFalse(any(e["event_type"] == "finalization_started" for e in events))
        self.assertTrue(any(e["event_type"] == "math_evidence_updated" for e in events))

    async def test_empty_ledger_finalization_retains_early_pure_reasoning(self):
        config = self.config()
        config["limits"]["max_model_turns"] = 2
        partial = "Definition: keep the original variable r=122/39.\n\n" + "Algebra.\n\n" * 300
        model = Model([FakeMessage(partial, finish_reason="length"), FakeMessage("#### 161")])
        output, events = await self.run_case(config, model)
        self.assertEqual(output.status, "success")
        self.assertIn("r=122/39", str(model.invocations[-1]["messages"]))
        self.assertEqual(output.metadata["finalization_context"]["mode"], "continue_history")
        self.assertFalse(model.invocations[-1]["tools"])
        self.assertTrue(any(e["event_type"] == "finalization_context" for e in events))

    async def test_final_request_reserves_output_within_cumulative_budget(self):
        config = self.config()
        config["limits"]["max_total_tokens"] = 18000
        first = FakeMessage("Definition: retain m=9, n=20.\n\n" + "Reasoning.\n\n" * 1000,
                            usage={"input_tokens": 1000, "output_tokens": 4000, "total_tokens": 5000},
                            finish_reason="length")
        model = Model([first, FakeMessage("#### 29")])
        output, _ = await self.run_case(config, model)
        self.assertEqual(output.status, "success")
        request = model.invocations[-1]
        prompt = int(1.25 * estimate_tokens([*request["messages"], {"tools": []}])) + 128
        cap = request["options"]["options"]["num_predict"]
        self.assertLessEqual(prompt + cap, 13000)

    async def test_unaffordable_protected_question_does_not_issue_a_request(self):
        config = self.config()
        profile = config["model"]["profiles"][config["model"]["default_profile"]]
        profile.update(context_window=256, max_tokens=64)
        model = Model([])
        output, _ = await self.run_case(config, model)
        self.assertEqual(output.error["type"], "token_budget")
        self.assertEqual(model.invocations, [])


if __name__ == "__main__":
    unittest.main()
