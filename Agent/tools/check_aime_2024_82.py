"""Reproduce a bounded, exhaustive check of AIME 2024:82."""
from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from Agent.tools.math_tools import execute_math_tool

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'runs/aime-qwen4b-same-error-3/bounded_enumeration_aime_2024_82.json'


def median(values: tuple[int, ...]) -> Fraction:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return Fraction(ordered[n // 2])
    return Fraction(ordered[n // 2 - 1] + ordered[n // 2], 2)


def valid(values: tuple[int, ...]) -> bool:
    counts = Counter(values)
    modes = [value for value, count in counts.items() if count == max(counts.values())]
    middle = median(values)
    return (sum(values) == 30 and modes == [9] and middle.denominator == 1
            and middle.numerator > 0 and middle.numerator not in values)


def constrained() -> tuple[list[tuple[int, ...]], int]:
    found: set[tuple[int, ...]] = set()
    checked = 0
    # At least two 9s are needed for a unique mode. Four 9s exceed the sum.
    for nines in (2, 3):
        remainder = 30 - 9 * nines
        # Unique mode: each other value may occur at most nines - 1 times.
        def visit(start: int, remaining: int, parts: tuple[int, ...]) -> None:
            nonlocal checked
            if remaining == 0:
                values = tuple(sorted((9,) * nines + parts))
                checked += 1
                if valid(values):
                    found.add(values)
                return
            for value in range(start, remaining + 1):
                if value == 9 or parts.count(value) >= nines - 1:
                    continue
                visit(value, remaining - value, parts + (value,))
        visit(1, remainder, ())
    return sorted(found), checked


def all_partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    for value in range(min(total, maximum or total), 0, -1):
        for rest in all_partitions(total - value, value):
            yield (value,) + rest


def main() -> None:
    bounded, checked = constrained()
    exhaustive = sorted(tuple(sorted(values)) for values in all_partitions(30) if valid(values))
    assert bounded == exhaustive, (bounded, exhaustive)
    output = {
        'problem_id': 'aime:2024:82', 'constrained_cases_checked': checked,
        'all_integer_partitions_checked': sum(1 for _ in all_partitions(30)),
        'valid_lists': [{'values': list(values), 'median': str(median(values)),
                         'sum_of_squares': sum(value * value for value in values)}
                        for values in bounded],
        'methods_agree': True,
    }
    recorded = ROOT / 'runs/aime-qwen4b-same-error-3/events.jsonl'
    event = next(row for line in recorded.open()
                 if (row := json.loads(line)).get('sample_id') == 'aime:2024:82'
                 and row.get('event_type') == 'tool_result' and row.get('sequence_number') == 56)
    original = event['payload']['call']['args']['code']
    old_check = 'isinstance(median, int) and median > 0 and median not in lst'
    assert original.count(old_check) == 2
    variants = {
        'median_only': 'median == int(median) and median > 0 and median not in lst',
        'median_and_unique_mode': 'modes == [9] and median == int(median) and median > 0 and median not in lst',
    }
    output['recorded_code_replays'] = {}
    for label, replacement in variants.items():
        result = execute_math_tool('python_calculate', {'code': original.replace(old_check, replacement)},
                                   timeout_seconds=5)
        payload = json.loads(result.output)
        assert result.success, payload.get('error')
        output['recorded_code_replays'][label] = [line for line in payload['stdout'].splitlines()
                                                  if line.startswith('VALID!')]
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
