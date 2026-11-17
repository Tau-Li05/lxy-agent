from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from Agent.engine import estimate_tokens
from Agent.runtime.math_workflow import MathEvidence
from Agent.tests import test_math_tool_workflow as workflow_tests
from Agent.tests.test_math_tool_workflow import Model, tool_message
from Agent.tests.test_runtime_v3 import FakeMessage
from Agent.tools.math_tools import MathToolSession, execute_math_tool


def call(code, call_id='c', purpose='calculate', constraints=None):
    args = {'code':code, 'purpose':purpose}
    if constraints is not None:
        args['constraints'] = constraints
    return {'id':call_id,'name':'python_calculate','args':args}


def observe(ledger, request):
    result = execute_math_tool(request['name'],request['args'])
    row = ledger.observe(request,result)
    return result,row


class EvidenceTests(unittest.TestCase):
    def test_enumeration_rejects_missing_or_unexecuted_assertions(self):
        for code,constraints in [('print(1)', ['count totals nine']), ('assert True\nprint(1)', []),
                                 ('if False:\n    assert True\nprint(1)', ['count totals nine'])]:
            request = call(code,purpose='enumerate',constraints=constraints)
            result = execute_math_tool(request['name'],request['args'])
            self.assertFalse(result.success)
            self.assertEqual(result.error['type'],'constraint_check_required')

    def test_original_population_assertion_catches_actual_regression(self):
        request = call('counts=(5,2,1)\nassert sum(counts)==9, "all nine players must be assigned"\nprint(5292)',
                       purpose='enumerate',constraints=['all nine players assigned'])
        result = execute_math_tool(request['name'],request['args'])
        self.assertEqual((result.error['type'],result.error['line']),('AssertionError',2))
        self.assertNotIn('5292',json.loads(result.output)['stdout'])

    def test_enumeration_tracks_executed_checks(self):
        request = call('import json\ncounts=(4,3,2)\nassert sum(counts)==9\nassert counts[0]>counts[1]>counts[2]>=1\n'
                       'print(json.dumps({"candidate_answer":16,"search_complete":True,"method":"enumeration"}))',
                       purpose='enumerate',constraints=['nine players','ordered nonempty counts'])
        ledger = MathEvidence()
        result,row = observe(ledger,request)
        self.assertTrue(result.success,result.output)
        self.assertEqual(row['assertions_passed'],2)
        candidate = ledger.snapshot()['candidates'][0]
        self.assertEqual(candidate['status'],'assertions_passed')
        self.assertTrue(candidate['search_complete_reported'])
        self.assertIn('do not prove',ledger.snapshot()['notice'])

    def test_partial_failed_candidate_not_promoted(self):
        request = call('import json\nprint(json.dumps({"candidate_answer":396}))\nassert False, "later check failed"')
        ledger = MathEvidence()
        result,row = observe(ledger,request)
        self.assertFalse(result.success)
        self.assertEqual(ledger.snapshot()['candidates'][0]['status'],'failed_partial')
        self.assertFalse(ledger.should_finalize)
        self.assertIn('396',row['fact_excerpt'])

    def test_bare_intermediate_and_exact_result_are_not_final_candidates(self):
        ledger = MathEvidence()
        observe(ledger,call('print(279)'))
        ledger.observe({'id':'e','name':'exact_calculate','args':{'expression':'9+18+252'}},
                       execute_math_tool('exact_calculate',{'expression':'9+18+252'}))
        self.assertEqual(ledger.snapshot()['candidates'],[])
        self.assertEqual(len(ledger.snapshot()['recent_results']),2)
        self.assertFalse(ledger.should_finalize)

    def test_conflicting_values_prevent_automatic_repetition_stop(self):
        ledger = MathEvidence()
        for value in [110,179]:
            observe(ledger,call(f'print(\'{{"candidate_answer":{value}}}\')',str(value)))
        ledger.cached_successes = 10
        self.assertTrue(ledger.snapshot()['conflicting_candidates'])
        self.assertFalse(ledger.should_finalize)

    def test_distinct_verification_methods_allowed_then_stop_repetition(self):
        ledger = MathEvidence()
        for i in range(3):
            # Different executable programs; equality is only agreement, never proof.
            observe(ledger,call(f'x={i}\nassert x>=0\nprint(\'{{"candidate_answer":393}}\')',str(i),'verify'))
            self.assertEqual(ledger.should_finalize,i==2)

    def test_success_cache_covers_parallel_requests_and_preserves_result(self):
        session = MathToolSession(cache_successes=True)
        with patch('Agent.tools.math_tools.execute_math_tool',wraps=execute_math_tool) as execute:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: session.execute('python_calculate',{'code':'print(393)'}),range(2)))
            self.assertEqual(execute.call_count,1)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(sum(bool(json.loads(r.output).get('cached_success')) for r in results),1)
        self.assertTrue(all(json.loads(r.output)['stdout']=='393\n' for r in results))
        # Independent code remains executable.
        result = session.execute('python_calculate',{'code':'assert 390+3==393\nprint(393)'})
        self.assertNotIn('cached_success',json.loads(result.output))

    def test_snapshot_and_final_prompt_are_bounded_and_keep_original_question(self):
        ledger = MathEvidence()
        for i in range(30):
            observe(ledger,call(f'print(\'{{"candidate_answer":{i}}}\')',str(i)))
        self.assertLessEqual(len(ledger.snapshot()['candidates']),12)
        self.assertLessEqual(len(ledger.snapshot()['recent_results']),8)
        messages = ledger.final_messages('Finish with #### integer.',[{'role':'user','content':'Original constraint: nine players.'}], 'x '*10000)
        self.assertIn('nine players',str(messages))
        self.assertEqual(ledger.finalization_info['mode'], 'continue_history')
        self.assertIn('conflicting_candidates',str(messages))
        self.assertLess(estimate_tokens(messages),5000)

    def test_truncated_output_does_not_create_candidates(self):
        ledger = MathEvidence()
        ledger.observe_model('#### 110',truncated=True)
        self.assertEqual(ledger.snapshot()['candidates'],[])
        ledger.observe_model('#### 110')
        self.assertEqual(ledger.snapshot()['candidates'][0]['status'],'proposed')

    def test_explicit_answer_with_agreeing_exact_result_gets_short_review(self):
        ledger = MathEvidence()
        ledger.observe_model('The number of ways is 902.')
        ledger.observe({'id':'exact','name':'exact_calculate','args':{'expression':'2+900'}},
                       execute_math_tool('exact_calculate',{'expression':'2+900'}))
        self.assertTrue(ledger.can_review_candidate)
        self.assertFalse(ledger.can_extract)
        messages = ledger.final_messages('Finish.', [{'role':'user','content':'Count the ways.'}], None)
        self.assertEqual(ledger.finalization_info['mode'], 'review_candidate')
        self.assertIn('original constraints', str(messages))
        ledger.observe_model('The answer is 901.')
        self.assertFalse(ledger.can_review_candidate)

    def test_bare_number_or_failed_calculation_never_triggers_review(self):
        ledger = MathEvidence()
        ledger.observe_model('We computed 902 in an intermediate step.')
        ledger.observe({'id':'exact','name':'exact_calculate','args':{'expression':'2+900'}},
                       execute_math_tool('exact_calculate',{'expression':'2+900'}))
        self.assertFalse(ledger.can_review_candidate)
        ledger.observe_model('The answer is 902.')
        ledger.observe({'id':'bad','name':'python_calculate','args':{'code':'assert False'}},
                       execute_math_tool('python_calculate',{'code':'assert False'}))
        self.assertFalse(ledger.can_review_candidate)

    def test_assumed_answer_is_not_a_closeout_signal(self):
        ledger = MathEvidence()
        ledger.observe_model("Let's assume the answer is 36. Final answer seems to be 36.")
        ledger.observe({'id':'exact','name':'exact_calculate','args':{'expression':'25+11'}},
                       execute_math_tool('exact_calculate',{'expression':'25+11'}))
        self.assertEqual(ledger.snapshot()['candidates'][0]['status'], 'tentative')
        self.assertFalse(ledger.can_review_candidate)


class EvidenceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    config = workflow_tests.MathWorkflowTests.config
    run_case = workflow_tests.MathWorkflowTests.run_case

    async def test_repeated_success_reuses_result_then_finalizes_with_evidence(self):
        config = self.config()
        config['features']['math_tools'].update(cache_successes=True, stop_on_repetition=True)
        code = 'import json\nassert 390+3==393\nprint(json.dumps({"candidate_answer":393,"method":"exact"}))'
        model = Model([tool_message('a',code),tool_message('b',code),tool_message('c',code),FakeMessage('#### 393')])
        with patch('Agent.tools.math_tools.execute_math_tool',wraps=execute_math_tool) as execute:
            result,events = await self.run_case(config,model)
        self.assertEqual(result.status,'success')
        self.assertEqual(execute.call_count,1)
        self.assertEqual(sum(e['event_type']=='tool_success_reused' for e in events),2)
        self.assertEqual(next(e for e in events if e['event_type']=='finalization_started')['payload']['reason'],'repeated_success')
        self.assertFalse(model.invocations[-1]['tools'])
        prompt = str(model.invocations[-1]['messages'])
        self.assertIn('393',prompt)
        self.assertIn('assertions_passed',prompt)
        self.assertFalse(any(isinstance(m, dict) and m.get('role') == 'tool' for m in model.invocations[-1]['messages']))
        self.assertEqual(result.metadata['math_evidence']['cached_successes'],2)

    async def test_finalization_compacts_history_and_keeps_failed_output_label(self):
        config = self.config()
        config['limits']['max_model_turns']=2
        code = 'print(\'{{"candidate_answer":396}}\')\nassert False'.replace('{{','{').replace('}}','}')
        model = Model([tool_message('a',code),FakeMessage('Candidate is unverified.\n#### 396')])
        result,events = await self.run_case(config,model)
        self.assertEqual(result.status,'success')
        prompt = str(model.invocations[-1]['messages'])
        self.assertIn('failed_partial',prompt)
        self.assertIn('Do not restart',prompt)
        self.assertIn('Compute 9+18+252.',prompt)
        self.assertTrue(any(e['event_type']=='finalization_evidence' for e in events))

    async def test_workflow_can_be_disabled_for_ablation(self):
        config=self.config()
        config['features']['math_tools']['workflow_enabled']=False
        model=Model([tool_message('a','print(279)'),tool_message('b','print(279)'),FakeMessage('#### 279')])
        with patch('Agent.tools.math_tools.execute_math_tool',wraps=execute_math_tool) as execute:
            result,events=await self.run_case(config,model)
        self.assertEqual(result.status,'success')
        self.assertEqual(execute.call_count,2)
        self.assertFalse(any(e['event_type']=='math_evidence_updated' for e in events))


if __name__=='__main__':
    unittest.main()
