from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from Agent.evaluation.config import load_run_config
from Agent.runtime.qwen_harness import QwenHarnessRuntime, _output_budget, _request_budget
from Agent.tests.test_runtime_v3 import FakeMessage
from Agent.tools.math_tools import MathToolSession, execute_math_tool, math_tool_schemas

ROOT = Path(__file__).resolve().parents[2]


class MathToolTests(unittest.TestCase):






    def test_exact_rationals_and_final_sum(self):
        for expression, expected in [('9+18+252', '279'), ('1/3+0.1', '13/30'),
                                     ('comb(8,4)', '70'), ('2**-3', '1/8'),
                                     ('gcd(18,24)+isqrt(81)', '15')]:
            result = execute_math_tool('exact_calculate', {'expression': expression})
            self.assertTrue(result.success, result.output)
            self.assertEqual(json.loads(result.output)['result'], expected)

    def test_exact_rejects_code_and_excessive_work(self):
        for expression in ["__import__('os')", '(1).__class__', '[x for x in range(10)]',
                           '2**100000000', 'factorial(100000)', '1e-100000000', 'True', '1/0']:
            self.assertFalse(execute_math_tool('exact_calculate', {'expression': expression}).success, expression)
        self.assertEqual(json.loads(execute_math_tool('exact_calculate', {'expression':'1/0'}).output)['error_type'], 'ZeroDivisionError')

    def test_structured_errors_and_bounded_output(self):
        result = json.loads(execute_math_tool('python_calculate', {'code': 'x = []\nprint(x[1])'}).output)
        self.assertEqual((result['error_type'], result['line']), ('IndexError', 2))
        self.assertFalse(result['timed_out'])
        result = json.loads(execute_math_tool('python_calculate', {'code': 'if :'}).output)
        self.assertEqual((result['error_type'], result['line']), ('SyntaxError', 1))
        result = json.loads(execute_math_tool('python_calculate', {'code': 'while True: pass'}, timeout_seconds=0.05).output)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['error_type'], 'timeout')
        self.assertIn('finite search bounds', result['message'])
        truncated = execute_math_tool('python_calculate', {'code': "print('a'*10000)"})
        result = json.loads(truncated.output)
        self.assertFalse(truncated.success)
        self.assertEqual(result['error_type'], 'output_truncated')
        self.assertTrue(result['execution_succeeded'])
        self.assertFalse(result['usable_result'])
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(result['stdout']), 4000)
        self.assertEqual(json.loads(execute_math_tool('python_calculate', {'code': 'import socket'}).output)['error_type'], 'unsafe_code')

    def test_truncated_output_is_deduplicated_until_code_changes(self):
        session = MathToolSession()
        code = "print('a'*10000)"
        self.assertEqual(session.execute('python_calculate', {'code': code}).error['type'], 'output_truncated')
        with patch('Agent.tools.math_tools.execute_math_tool', side_effect=AssertionError('must not rerun')):
            repeat = session.execute('python_calculate', {'code': code})
        self.assertEqual(repeat.error['type'], 'duplicate_failed_call')
        self.assertTrue(session.execute('python_calculate', {'code': "print('match: 236')"}).success)

    def test_single_python_entry_and_legacy_alias(self):
        for mode in ['python', 'python_sympy', 'python_exact']:
            self.assertEqual([s['name'] for s in math_tool_schemas(mode)], ['python_calculate', 'exact_calculate'])
        self.assertTrue(execute_math_tool('sympy_calculate', {'code': 'print(42)'}).success)
        self.assertEqual(math_tool_schemas('none'), [])

    def test_empty_stdout_is_unusable_and_same_call_is_blocked(self):
        session = MathToolSession()
        code = "import sympy as sp\nroots = sp.nroots(sp.Symbol('x')**2-1)\nroots"
        first = session.execute('python_calculate', {'code': code})
        payload = json.loads(first.output)
        self.assertFalse(first.success)
        self.assertEqual(payload['error_type'], 'empty_output')
        self.assertTrue(payload['execution_succeeded'])
        self.assertFalse(payload['usable_result'])
        self.assertIn('print(roots)', payload['message'])
        with patch('Agent.tools.math_tools.execute_math_tool', side_effect=AssertionError('must not rerun')):
            duplicate = session.execute('python_calculate', {'code': code})
        repeated = json.loads(duplicate.output)
        self.assertEqual(repeated['error_type'], 'duplicate_failed_call')
        self.assertEqual(repeated['previous_error']['type'], 'empty_output')
        self.assertEqual(repeated['previous_result']['error_type'], 'empty_output')
        corrected = session.execute('python_calculate', {'code': code + '\nprint(roots)'})
        self.assertTrue(corrected.success)
        self.assertEqual(json.loads(corrected.output)['usable_result'], True)

    def test_failed_calls_deduplicate_across_aliases_and_concurrency(self):
        session = MathToolSession()
        with patch('Agent.tools.math_tools.execute_math_tool', wraps=execute_math_tool) as execute:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: session.execute('python_calculate', {'code': 'print(1/0)'}), range(2)))
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(sum(r.error['type'] == 'duplicate_failed_call' for r in results), 1)
            self.assertEqual(session.execute('sympy_calculate', {'code': '# same\nprint(1 / 0)'}).error['type'], 'duplicate_failed_call')
            self.assertEqual(execute.call_count, 1)
            self.assertTrue(session.execute('python_calculate', {'code': 'print(1/2)'}).success)
            self.assertTrue(session.execute('python_calculate', {'code': 'print(1/2)'}).success)
            self.assertEqual(execute.call_count, 3)
        self.assertNotEqual(MathToolSession().execute('python_calculate', {'code': 'print(1/0)'}).error['type'], 'duplicate_failed_call')

    def test_same_type_error_diagnostic_preserves_original_error(self):
        session = MathToolSession()
        first = session.execute('python_calculate', {'code': 'center = 1.0\nprint(center[0])'})
        second = session.execute('python_calculate', {'code': 'center = 2.0\nprint(center[1])'})
        self.assertEqual(first.error['type'], 'TypeError')
        self.assertEqual(second.error['type'], 'TypeError')
        diagnostic = json.loads(second.output)['same_error_diagnostic']
        self.assertEqual(diagnostic['streak'], 2)
        self.assertEqual(diagnostic['failing_line'], 'print(center[1])')
        self.assertEqual(diagnostic['suggested_probes'], ['print(type(center), repr(center))'])
        self.assertTrue(session.execute('python_calculate', {'code': 'print(279)'}).success)
        third = session.execute('python_calculate', {'code': 'center = 3.0\nprint(center[2])'})
        self.assertNotIn('same_error_diagnostic', json.loads(third.output))


class Model:
    def __init__(self, responses):
        self.responses, self.invocations = list(responses), []
    def bind_tools(self, tools):
        return View(self, tools, {})
    def bind(self, **options):
        return View(self, [], options)
    async def ainvoke(self, messages):
        return await View(self, [], {}).ainvoke(messages)


class View:
    def __init__(self, model, tools, options):
        self.model, self.tools, self.options = model, tools, options
    def bind(self, **options):
        return View(self.model, self.tools, {**self.options, **options})
    async def ainvoke(self, messages):
        self.model.invocations.append({'tools': self.tools, 'options': self.options, 'messages': list(messages)})
        if not self.model.responses:
            raise AssertionError('unexpected model invocation')
        return self.model.responses.pop(0)


def tool_message(call_id, code):
    return FakeMessage('', [{'id': call_id, 'name': 'python_calculate', 'args': {'code':code}}])


class MathWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        config = load_run_config(ROOT/'Agent/configs/v3/aime_qwen_harness.yaml', ROOT/'Agent')
        config['features']['difficulty_router'] = {'enabled':False}
        config['features']['planning'] = {'enabled':False}
        config['features']['parallel_tools'] = {'enabled':True, 'max_concurrency':2}
        return config

    async def run_case(self, config, model):
        item = {'id':'test:1', 'metadata':{'benchmark':'aime'}, 'environment':{},
                'turns':[[{'role':'user','content':'Compute 9+18+252.'}]], 'tools':[]}
        with tempfile.TemporaryDirectory() as tmp:
            runtime = QwenHarnessRuntime(config, ROOT, model=model)
            runtime.run_state_root = Path(tmp)
            result = await runtime.run(item)
            events = [json.loads(l) for l in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        return result, events

    async def test_pre_final_audit_is_opt_in_and_precedes_next_model_turn(self):
        config = self.config()
        config['features']['math_tools']['pre_final_audit'] = True
        model = Model([tool_message('a', 'print(279)'), FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertEqual(sum(e['event_type'] == 'pre_final_audit_prompted' for e in events), 1)
        self.assertIn('exact quantity and answer format', model.invocations[0]['messages'][0]['content'])
        self.assertIn('audit the work so far', model.invocations[1]['messages'][-1]['content'])
        self.assertTrue(model.invocations[1]['tools'])

        config['features']['math_tools']['pre_final_audit'] = False
        model = Model([tool_message('a', 'print(279)'), FakeMessage('#### 279')])
        _, events = await self.run_case(config, model)
        self.assertFalse(any(e['event_type'] == 'pre_final_audit_prompted' for e in events))
        self.assertNotIn('audit the work so far', str(model.invocations[1]['messages'][-1]))





    async def test_concise_progress_is_ephemeral_and_opt_in(self):
        config = self.config()
        config['features']['math_tools']['concise_progress'] = True
        model = Model([tool_message('a', 'print(279)'), FakeMessage('#### 279')])
        result, _ = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        for invocation in model.invocations:
            prompts = [str(message.get('content', '')) for message in invocation['messages']
                       if isinstance(message, dict) and message.get('role') == 'user']
            self.assertEqual(sum('Progress rule for this turn' in value for value in prompts), 1)
        self.assertEqual(config['features']['finalization']['max_tokens'], 2048)

        config['features']['math_tools']['concise_progress'] = False
        model = Model([tool_message('a', 'print(279)'), FakeMessage('#### 279')])
        await self.run_case(config, model)
        self.assertNotIn('Progress rule for this turn', str(model.invocations[0]['messages']))

    async def test_bounded_nonfinal_turn_continues_before_finalization(self):
        config = self.config()
        config['features']['math_tools'].update({
            'bounded_turns': True, 'bounded_turn_max_tokens': 4096,
            'bounded_turn_max_continuations': 2,
        })
        model = Model([FakeMessage('Partial work', finish_reason='length'),
                       FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertEqual(model.invocations[0]['options']['options']['num_predict'], 4096)
        self.assertEqual(model.invocations[1]['options']['options']['num_predict'], 4096)
        self.assertTrue(model.invocations[1]['tools'])
        self.assertEqual(sum(e['event_type'] == 'bounded_turn_continued' for e in events), 1)
        self.assertFalse(any(e['event_type'] == 'finalization_started' for e in events))
        self.assertIn('previous non-final response reached', str(model.invocations[1]['messages']))

    async def test_bounded_continuations_are_limited_and_final_cap_is_preserved(self):
        config = self.config()
        config['features']['math_tools'].update({
            'bounded_turns': True, 'bounded_turn_max_tokens': 4096,
            'bounded_turn_max_continuations': 2,
        })
        model = Model([FakeMessage('Step one', finish_reason='length'),
                       FakeMessage('Step two', finish_reason='length'),
                       FakeMessage('Step three', finish_reason='length'),
                       FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertEqual(sum(e['event_type'] == 'bounded_turn_continued' for e in events), 2)
        self.assertEqual([m['options']['options']['num_predict'] for m in model.invocations],
                         [4096, 4096, 4096, 2048])
        self.assertFalse(model.invocations[-1]['tools'])
        self.assertEqual(next(e for e in events if e['event_type'] == 'finalization_started')
                         ['payload']['reason'], 'output_limit')

    async def test_truncated_tool_call_is_not_executed_during_bounded_continuation(self):
        config = self.config()
        config['features']['math_tools'].update({
            'bounded_turns': True, 'bounded_turn_max_tokens': 4096,
            'bounded_turn_max_continuations': 2,
        })
        first = tool_message('a', 'print(279)')
        first.response_metadata = {'finish_reason': 'length'}
        model = Model([first, FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertFalse(any(e['event_type'] == 'tool_result' for e in events))
        tool_feedback = [json.loads(m['content']) for m in result.messages if m.get('role') == 'tool']
        self.assertEqual(tool_feedback[0]['error_type'], 'truncated_tool_call')

    async def test_duplicate_errors_reach_model_and_fixed_code_runs(self):
        config = self.config()
        config['features']['math_tools']['repair_turn_enabled'] = True
        first = FakeMessage('', [
            {'id': 'a', 'name': 'python_calculate', 'args': {'code':'print(1/0)'}},
            {'id': 'b', 'name': 'python_calculate', 'args': {'code':'print(1 / 0)'}},
        ])
        model = Model([first, tool_message('c','# repeated\nprint(1/0)'),
                       FakeMessage('The division by zero failed; I will compute a different expression.'),
                       tool_message('d','print(279)'), FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        tool_results = [json.loads(m['content']) for m in result.messages if m.get('role') == 'tool']
        self.assertEqual(sum(m.get('error_type')=='duplicate_failed_call' for m in tool_results), 1)
        self.assertEqual(sum(m.get('error_type')=='repair_required' for m in tool_results), 1)
        self.assertEqual(sum(e['event_type']=='tool_failure_deduplicated' for e in events), 2)
        self.assertEqual(sum(e['event_type']=='tool_repair_completed' for e in events), 1)
        self.assertFalse(model.invocations[2]['tools'])
        self.assertTrue(model.invocations[3]['tools'])
        self.assertEqual(result.final_answer, '#### 279')

    async def test_same_code_after_repair_remains_blocked(self):
        config = self.config()
        config['features']['math_tools']['repair_turn_enabled'] = True
        code = 'print(1/0)'
        model = Model([tool_message('a', code), tool_message('b', code),
                       tool_message('c', code), FakeMessage('I will change the failing expression.'),
                       tool_message('d', code), tool_message('e', 'print(279)'), FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        feedback = [json.loads(message['content']) for message in result.messages if message.get('role') == 'tool']
        self.assertEqual([row.get('error_type') for row in feedback[:4]],
                         ['ZeroDivisionError', 'duplicate_failed_call', 'repair_required', 'duplicate_failed_call'])
        self.assertFalse(model.invocations[3]['tools'])
        self.assertEqual(model.invocations[3]['options']['options']['num_predict'], 512)
        self.assertTrue(model.invocations[4]['tools'])
        self.assertEqual(sum(e['event_type'] == 'tool_repair_requested' for e in events), 1)

    async def test_default_skips_repair_turn_and_emits_same_error_diagnostic(self):
        first_code = 'center = 1.0\nprint(center[0])'
        second_code = 'center = 2.0\nprint(center[1])'
        model = Model([tool_message('a', first_code), tool_message('b', second_code),
                       FakeMessage('#### 279')])
        result, events = await self.run_case(self.config(), model)
        self.assertEqual(result.status, 'success')
        self.assertFalse(any(event['event_type'] == 'tool_repair_requested' for event in events))
        diagnostic = next(event for event in events if event['event_type'] == 'tool_same_error_diagnostic')
        self.assertEqual(diagnostic['payload']['suggested_probes'],
                         ['print(type(center), repr(center))'])
        self.assertTrue(model.invocations[2]['tools'])

    async def test_empty_output_feedback_reaches_model_and_corrected_call_runs(self):
        code = "import sympy as sp\nroots = sp.nroots(sp.Symbol('x')**2-1)\nroots"
        model = Model([tool_message('a', code), tool_message('b', code),
                       tool_message('c', code + '\nprint(roots)'), FakeMessage('#### 279')])
        result, events = await self.run_case(self.config(), model)
        self.assertEqual(result.status, 'success')
        feedback = [json.loads(message['content']) for message in result.messages if message.get('role') == 'tool']
        self.assertEqual([row.get('error_type') for row in feedback[:2]],
                         ['empty_output', 'duplicate_failed_call'])
        self.assertTrue(feedback[2]['success'])
        self.assertEqual(sum(event['event_type'] == 'tool_failure_deduplicated' for event in events), 1)

    async def test_token_reserve_finishes_before_limit_with_request_cap(self):
        config = self.config()
        config['limits']['max_total_tokens'] = 20000
        config['features']['finalization']['max_tokens'] = 2048
        first = tool_message('a',"print('x '*1800)")
        # Represents a substantial first reply plus input, still below its request allowance.
        first.usage_metadata = {'input_tokens': 2000, 'output_tokens': 7000, 'total_tokens': 9000}
        model = Model([first, FakeMessage('#### 279')])
        # A long tool result/history drives the next prompt estimate high enough to reserve finalization.
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertGreaterEqual(model.invocations[0]['options']['options']['num_predict'], 7000)
        self.assertEqual(model.invocations[-1]['options']['options']['num_predict'], 2048)
        self.assertFalse(model.invocations[-1]['tools'])
        final_event = next(e for e in events if e['event_type']=='finalization_started')
        self.assertEqual(final_event['payload']['reason'], 'token_reserve')
        self.assertLess(result.usage['total_tokens'],20000)


    async def test_partial_response_uses_reserved_final_turn(self):
        config = self.config()
        config['features']['verification'] = {'enabled':True, 'method':'self'}
        model = Model([FakeMessage('Partial derivation', finish_reason='length'), FakeMessage('#### 279')])
        result, events = await self.run_case(config, model)
        self.assertEqual(result.status, 'success')
        self.assertEqual(len(model.invocations),2)
        self.assertFalse(model.invocations[-1]['tools'])
        self.assertEqual(next(e for e in events if e['event_type']=='finalization_started')['payload']['reason'],'output_limit')

    async def test_tool_budget_rejection_still_gets_final_answer(self):
        config = self.config()
        config['limits']['max_tool_calls'] = 1
        first = FakeMessage('', [
            {'id':'a','name':'exact_calculate','args':{'expression':'1+1'}},
            {'id':'b','name':'exact_calculate','args':{'expression':'2+2'}},
        ])
        model = Model([first, FakeMessage('#### 279')])
        result,events = await self.run_case(config,model)
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.metadata['tool_call_count'],0)
        self.assertFalse(model.invocations[-1]['tools'])
        self.assertTrue(any(e['event_type']=='tool_batch_budget_rejected' for e in events))

    async def test_exhausted_budget_does_not_issue_more_requests(self):
        config = self.config()
        config['limits']['max_total_tokens'] = 1
        model = Model([])
        result,_ = await self.run_case(config,model)
        self.assertEqual(result.error['type'], 'token_budget')
        self.assertEqual(len(model.invocations),0)

    def test_budget_estimate_reserves_next_input_and_output(self):
        messages = [{'role':'user','content':'Compute 9+18+252.'}]
        normal, final = _request_budget(messages, math_tool_schemas('python_exact'), 65536,16384,2048)
        self.assertEqual(normal,16384)
        self.assertEqual(final,2048)
        normal, final = _request_budget(messages, math_tool_schemas('python_exact'),3000,16384,2048)
        self.assertLess(normal,2048)
        self.assertGreater(final,0)
        self.assertEqual(_request_budget(messages,[],1,16384,2048),(0,0))

    def test_openai_request_limit_uses_max_tokens(self):
        model = Model([])
        bound = _output_budget(model, {'provider':'openai_compatible'},1234)
        self.assertEqual(bound.options, {'max_tokens':1234})

    def test_ollama_limits_preserve_existing_options(self):
        model = Model([])
        model.temperature = 0.2
        model.top_k = 30
        model.kwargs = {'options': {'repeat_penalty': 1.1}}
        profile = {'provider':'ollama','context_window':65536,'temperature':0,'seed':42}
        bound = _output_budget(model, profile,1234)
        self.assertEqual(bound.options, {'options': {'num_ctx':65536,'temperature':0.2,'top_k':30,
                                                    'repeat_penalty':1.1,'seed':42,'num_predict':1234}})


if __name__ == '__main__':
    unittest.main()
