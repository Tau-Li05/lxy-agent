from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Callable


def _quotas(sizes: dict[tuple[str, ...], int], total: int) -> dict[tuple[str, ...], int]:
    population = sum(sizes.values())
    raw = {key: sizes[key] * total / population for key in sizes}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def stratified_split(
    records: list[dict[str, Any]],
    stratum: Callable[[dict[str, Any]], tuple[str, ...]],
    key: Callable[[dict[str, Any]], str],
    seed: int,
    validation_count: int,
    test_count: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[stratum(record)].append(record)
    sizes = {name: len(items) for name, items in groups.items()}
    test_quotas = _quotas(sizes, test_count)
    validation_quotas = _quotas(sizes, validation_count)
    output = {"train": [], "validation": [], "test": []}
    for name in sorted(groups):
        items = sorted(
            groups[name],
            key=lambda item: hashlib.sha256(f"{seed}:{key(item)}".encode()).hexdigest(),
        )
        test_end = test_quotas[name]
        validation_end = test_end + validation_quotas[name]
        if validation_end > len(items):
            raise ValueError(f"Stratum {name} is too small for requested quotas")
        output["test"].extend(items[:test_end])
        output["validation"].extend(items[test_end:validation_end])
        output["train"].extend(items[validation_end:])
    for split in output:
        output[split].sort(key=key)
    return output

