from __future__ import annotations

from typing import Any, Iterable


def partition_metrics(results: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Separate development metrics from final-test metrics and reject semantic mixing."""
    output: dict[str, list[dict[str, Any]]] = {"dev_metrics": [], "test_metrics": []}
    for result in results:
        scope = result.get("evaluation_scope")
        split = result.get("split")
        if scope == "development":
            if split != "validation":
                raise ValueError("development metrics must come from validation split")
            output["dev_metrics"].append(result)
        elif scope == "final_test":
            if split != "test":
                raise ValueError("final test metrics must come from test split")
            output["test_metrics"].append(result)
        else:
            raise ValueError(f"unknown evaluation_scope: {scope!r}")
    return output

