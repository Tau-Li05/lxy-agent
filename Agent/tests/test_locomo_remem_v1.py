import asyncio
import unittest

from Agent.runtime.locomo_remem_v1 import EpisodicMemoryV1


ITEM = {
    "metadata": {"conversation_id": "conv-test"},
    "turns": [[{"role": "user", "content":
               "SESSION 1\nDATE: 8 May 2023\n[D1:1] Caroline: I visited a support group yesterday.\n"
               "[D1:2] Melanie: That sounds helpful."}]],
    "environment": {"questions": [{"id": "q1", "question": "When did Caroline visit?"}]},
}


class FakeResponse:
    content = ('{"gists":[{"text":"Caroline visited a support group",'
               '"source_ids":["D1:1"]}],"facts":[{"subject":"Caroline",'
               '"predicate":"visited","object":"support group","time":"7 May 2023",'
               '"source_ids":["D1:1"]}]}')
    usage_metadata = {"input_tokens": 20, "output_tokens": 15}


class FakeModel:
    async def ainvoke(self, messages):
        assert "When did Caroline visit?" not in messages[0]["content"]
        return FakeResponse()


class EpisodicMemoryTests(unittest.TestCase):
    def test_source_and_ablation(self):
        async def check():
            full = {"mode": "remem_v1", "use_vector": False, "use_graph": True,
                    "use_raw_backtrace": True, "use_time": True}
            memory = EpisodicMemoryV1(ITEM, full, FakeModel())
            await memory.build()
            context, stats = memory.retrieve("When did Caroline visit the support group?")
            self.assertIn("7 May 2023", context)
            self.assertIn("[D1:1] Caroline:", context)
            self.assertEqual(stats["raw_backtraced"], 1)
            self.assertEqual(memory.write_calls, 1)
            self.assertEqual(memory.write_prompt_tokens, 20)
            no_raw = EpisodicMemoryV1(ITEM, {**full, "use_raw_backtrace": False}, FakeModel())
            await no_raw.build()
            context, stats = no_raw.retrieve("When did Caroline visit the support group?")
            self.assertNotIn("[D1:1] Caroline:", context)
            self.assertEqual(stats["raw_backtraced"], 0)
            no_time = EpisodicMemoryV1(ITEM, {**full, "use_time": False}, FakeModel())
            await no_time.build()
            context, _ = no_time.retrieve("When did Caroline visit the support group?")
            self.assertNotIn("7 May 2023", context)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
