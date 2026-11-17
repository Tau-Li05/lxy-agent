"""Compare the three Ollama tool-feedback cases with the recorded baseline run."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

IDS = ('aime:2024:82', 'aime:2025:19', 'aime:2026:23')
ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'runs/aime-qwen4b-tools-current-full'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(run: Path) -> dict[str, dict]:
    results = {row['id']: row for row in read_jsonl(run / 'sample_results.jsonl')}
    trajectories = {row['sample_id']: row for row in read_jsonl(run / 'trajectories.jsonl')}
    events = read_jsonl(run / 'events.jsonl')
    rows = {}
    for sid in IDS:
        if sid not in results or sid not in trajectories:
            raise ValueError(f'{run}: missing {sid}')
        trajectory_id = trajectories[sid]['trajectory_id']
        matched = [event for event in events if event['sample_id'] == sid
                   and event['trajectory_id'] == trajectory_id]
        tools = [event['payload']['result'] for event in matched
                 if event['event_type'] == 'tool_result']
        error_types = Counter()
        nonempty_stdout = 0
        empty_stdout_successes = 0
        for result in tools:
            error = result.get('error') or {}
            if error:
                error_types[error.get('type', 'unknown')] += 1
            elif result.get('success'):
                raw = result.get('output', '')
                if result.get('artifact'):
                    with gzip.open(run / result['artifact']['path'], 'rt') as source:
                        raw = json.load(source).get('output', '')
                try:
                    stdout = json.loads(raw).get('stdout', '')
                except (TypeError, ValueError, AttributeError):
                    stdout = ''
                if stdout.strip():
                    nonempty_stdout += 1
                else:
                    empty_stdout_successes += 1
        final = [event for event in matched if event['event_type'] == 'finalization_started']
        rows[sid] = {
            'status': results[sid]['status'],
            'prediction': results[sid].get('prediction'),
            'reference': results[sid].get('reference'),
            'tool_calls': len(tools),
            'nonempty_stdout_results': nonempty_stdout,
            'empty_stdout_successes': empty_stdout_successes,
            'errors': dict(error_types),
            'finalization_reason': final[-1]['payload']['reason'] if final else 'none',
            'total_tokens': trajectories[sid].get('usage', {}).get('total_tokens'),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='runs/<run-name> created by Agent.cli.evaluate')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    baseline, trial = load(BASELINE), load(run)
    lines = [
        '# Three-case tool-feedback comparison', '',
        f'Baseline: `{BASELINE}`', f'Trial: `{run}`', '',
        '| Sample | Baseline status | Trial status | Baseline tool errors | Trial tool errors | Baseline empty stdout | Trial empty stdout | Baseline nonempty stdout | Trial nonempty stdout | Trial answer |',
        '|---|---|---|---|---|---:|---:|---:|---:|---|',
    ]
    for sid in IDS:
        a, b = baseline[sid], trial[sid]
        lines.append(f"| {sid} | {a['status']} | {b['status']} | {a['errors']} | {b['errors']} | "
                     f"{a['empty_stdout_successes']} | {b['empty_stdout_successes']} | "
                     f"{a['nonempty_stdout_results']} | {b['nonempty_stdout_results']} | {b['prediction']} |")
    lines += ['', '## Per-case details', '']
    for sid in IDS:
        lines += [f'### {sid}', '', '```json', json.dumps({'baseline': baseline[sid], 'trial': trial[sid]},
                    ensure_ascii=False, indent=2), '```', '']
    lines += ['This is one run per case. Inspect the trial events and final answer before attributing any answer change to the tool rules.']
    report = run / 'tool_feedback_comparison.md'
    report.write_text('\n'.join(lines) + '\n')
    print(report)


if __name__ == '__main__':
    main()
