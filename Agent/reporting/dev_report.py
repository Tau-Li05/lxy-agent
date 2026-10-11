from __future__ import annotations

from typing import Any

from Agent.data_pipeline.reporting import partition_metrics
from Agent.evaluation.models import BenchmarkMetrics


def build_dev_report(metrics: BenchmarkMetrics) -> dict[str, Any]:
    value = metrics.to_dict()
    partitioned = partition_metrics([value])
    if partitioned["test_metrics"]:
        raise ValueError("development metrics cannot be written as test_metrics")
    return {"schema_version": "1.0", "evaluation_scope": "development", "dev_metrics": partitioned["dev_metrics"]}
