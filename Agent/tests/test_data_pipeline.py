from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Agent.data_pipeline.adapters import ADAPTERS
from Agent.data_pipeline.builder import _load_config, discover_configs
from Agent.data_pipeline.extractors import extract_boxed, extract_gsm8k, livebench_answer
from Agent.data_pipeline.io import read_jsonl, stable_json, write_jsonl
from Agent.data_pipeline.reporting import partition_metrics
from Agent.data_pipeline.splitting import stratified_split
from Agent.data_pipeline.validation import ensure_policy_isolated, ensure_same_ids


ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = ROOT / "Agent"
CONFIG_DIR = AGENT_ROOT / "configs" / "conversions"


class ExtractorTests(unittest.TestCase):
    def test_boxed_nested_and_unbraced(self) -> None:
        self.assertEqual(extract_boxed(r"x=\boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(extract_boxed(r"x=\boxed 9."), "9")
        self.assertIsNone(extract_boxed(r"x=\boxed{}"))

    def test_gsm8k_uses_last_marker(self) -> None:
        self.assertEqual(extract_gsm8k("#### 1\nwork\n#### 2"), "2")

    def test_livebench_preserves_non_list_math(self) -> None:
        self.assertEqual(livebench_answer(" 1, 06,3 "), "1,6,3")
        self.assertEqual(livebench_answer(r" \frac{1}{2} "), r"\frac{1}{2}")


class IOTests(unittest.TestCase):
    def test_jsonl_reader_does_not_split_unicode_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text('{"text":"a\u2028b"}\n', encoding="utf-8")
            rows = list(read_jsonl(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1]["text"], "a\u2028b")

    def test_stable_writer_is_byte_deterministic(self) -> None:
        records = [{"z": 1, "a": "中文"}]
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            self.assertEqual(write_jsonl(left, records), write_jsonl(right, records))
            self.assertEqual(left.read_bytes(), right.read_bytes())


class SplitTests(unittest.TestCase):
    def test_exact_stratified_counts_and_determinism(self) -> None:
        rows = [{"id": str(i), "group": str(i % 5)} for i in range(100)]
        kwargs = dict(stratum=lambda x: (x["group"],), key=lambda x: x["id"], seed=42, validation_count=5, test_count=5)
        first = stratified_split(rows, **kwargs)
        second = stratified_split(list(reversed(rows)), **kwargs)
        self.assertEqual({k: len(v) for k, v in first.items()}, {"train": 90, "validation": 5, "test": 5})
        self.assertEqual(first, second)


class IsolationTests(unittest.TestCase):
    def test_policy_rejects_reference(self) -> None:
        with self.assertRaises(ValueError):
            ensure_policy_isolated([{"id": "x", "input": {}, "reference": "42"}])

    def test_id_sets_must_match(self) -> None:
        with self.assertRaises(ValueError):
            ensure_same_ids([{"id": "a"}], [{"id": "b"}], "rl")

    def test_validation_cannot_be_reported_as_final_test(self) -> None:
        with self.assertRaises(ValueError):
            partition_metrics([{"evaluation_scope": "final_test", "split": "validation", "accuracy": 1.0}])
        separated = partition_metrics([
            {"evaluation_scope": "development", "split": "validation", "accuracy": 0.8},
            {"evaluation_scope": "final_test", "split": "test", "accuracy": 0.7},
        ])
        self.assertEqual(len(separated["dev_metrics"]), 1)
        self.assertEqual(len(separated["test_metrics"]), 1)


class RealDatasetAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configs = discover_configs(CONFIG_DIR, None)
        cls.results = {}
        for path, config in configs:
            cls.results[config["benchmark"]] = ADAPTERS[config["adapter"]](config, path, ROOT)

    def test_expected_canonical_counts(self) -> None:
        expected = {"competition_math": 12494, "aime24": 30, "aime25": 30, "aime26": 30, "gsm8k": 8792, "livebench_math": 368}
        self.assertEqual({key: len(value.records) for key, value in self.results.items()}, expected)

    def test_competition_split_and_rejections(self) -> None:
        result = self.results["competition_math"]
        counts = {split: sum(record["split"] == split for record in result.records) for split in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 11244, "validation": 625, "test": 625})
        reasons = {}
        for item in result.rejected:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        self.assertEqual(reasons, {"invalid_level": 2, "duplicate_question": 2, "answer_extraction_failed": 2})

    def test_ids_are_globally_unique(self) -> None:
        ids = [record["id"] for result in self.results.values() for record in result.records]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()

