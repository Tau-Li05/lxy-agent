"""Compare the 11 historical output-limit cases with the current tool-feedback run."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'runs/aime-qwen4b-resum-working-memory-closeout-full'

def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial_dir', type=Path)
    args = parser.parse_args()
    trial = args.trial_dir.resolve()
    old = {x['id']: x for x in rows(BASELINE / 'sample_results.jsonl')}
    new = {x['id']: x for x in rows(trial / 'sample_results.jsonl')}
    ids = [x['id'] for x in rows(BASELINE / 'sample_results.jsonl') if x['status'] == 'limit_exceeded']
    missing = set(ids) - set(new)
    if missing:
        raise ValueError(f'missing trial results: {sorted(missing)}')
    events = rows(trial / 'events.jsonl')
    trajectories = {x['sample_id']: x for x in rows(trial / 'trajectories.jsonl')}
    lines = ['# 11 historical output-limit cases: current tool-feedback trial', '',
             f'Historical run: `{BASELINE}`', f'Trial run: `{trial}`', '',
             '| Sample | Historical | Trial | Trial answer | Tool errors | Same-error diagnostics | Finalization reason |',
             '|---|---|---|---:|---|---:|---|']
    counts = Counter()
    for sid in ids:
        result = new[sid]
        matched = [x for x in events if x['sample_id'] == sid
                   and x['trajectory_id'] == trajectories[sid]['trajectory_id']]
        errors = Counter((x['payload']['result'].get('error') or {}).get('type')
                         for x in matched if x['event_type'] == 'tool_result'
                         and x['payload']['result'].get('error'))
        diagnostics = sum(x['event_type'] == 'tool_same_error_diagnostic' for x in matched)
        finals = [x['payload']['reason'] for x in matched if x['event_type'] == 'finalization_started']
        lines.append(f"| {sid} | {old[sid]['status']} | {result['status']} | "
                     f"{result.get('prediction', '—')} | {dict(errors)} | {diagnostics} | "
                     f"{finals[-1] if finals else 'none'} |")
        counts[result['status']] += 1
    lines += ['', f"Outcome counts: {dict(counts)}", '',
              'An incorrect answer is not counted as a success merely because it avoided truncation.',
              'This is a selected-failure comparison with a different configuration; it is not a paired full-run ablation.', '']
    report = trial / 'tool_feedback_11_comparison.md'
    report.write_text('\n'.join(lines))
    print(report)

if __name__ == '__main__':
    main()
