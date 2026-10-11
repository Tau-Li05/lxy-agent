from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from .models import BenchmarkMetrics, SampleResult


class CompetitionMathAggregator:
    def aggregate(self, results: Sequence[SampleResult]) -> BenchmarkMetrics:
        if not results:
            raise ValueError("cannot aggregate an empty result set")
        for result in results:
            if result.evaluation_scope != "development" or result.split != "validation":
                raise ValueError("competition_math DEV metrics require development/validation results")
        return BenchmarkMetrics(
            schema_version="1.0",
            benchmark="competition_math",
            evaluation_scope="development",
            split="validation",
            total=len(results),
            correct=sum(result.correct for result in results),
            accuracy=sum(result.correct for result in results) / len(results),
            status_counts=dict(sorted(Counter(result.status for result in results).items())),
            by_level=_group(results, lambda item: str(item.metadata.get("level", "unknown"))),
            by_category=_group(results, lambda item: str(item.metadata.get("category", "unknown"))),
            by_level_category=_group(
                results,
                lambda item: f"{item.metadata.get('level', 'unknown')} | {item.metadata.get('category', 'unknown')}",
            ),
        )


def _group(results: Sequence[SampleResult], key_fn: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[SampleResult]] = defaultdict(list)
    for result in results:
        groups[key_fn(result)].append(result)
    return {
        key: {
            "total": len(items),
            "correct": sum(item.correct for item in items),
            "accuracy": sum(item.correct for item in items) / len(items),
            "status_counts": dict(sorted(Counter(item.status for item in items).items())),
        }
        for key, items in sorted(groups.items())
    }
