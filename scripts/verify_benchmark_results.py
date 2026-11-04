#!/usr/bin/env python3
"""Compare a completed hologram-to-evaluation run with the published results."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from reproduce import compare

REPO = Path(__file__).resolve().parents[1]
METHODS = ['ours', 'depth_free', 'minip2d']


def metric_values(metrics):
    values = {key: metrics[key] for key in
              ['event_ap', 'maximum_f1', 'maximum_recall', 'truth_events', 'predicted_events']}
    for target in ['0.80', '0.90']:
        point = metrics[f'operating_at_recall_{target}']
        for key in ['precision', 'recall', 'false_positive_events']:
            values[f'{key}_at_recall_{target}'] = None if point is None else point[key]
    return values


def verify(run_root):
    report = {'status': 'PASS', 'metric_atol': 1e-12, 'methods': {}}
    for method in METHODS:
        actual = run_root / 'results' / method
        reference = REPO / 'reference/evaluation' / method
        missing = [name for name in ['metrics.json', 'predictions.parquet'] if not (actual / name).is_file()]
        if missing:
            report['methods'][method] = {'status': 'MISSING', 'missing': missing}
            report['status'] = 'MISMATCH'
            continue
        measured = metric_values(json.loads((actual / 'metrics.json').read_text()))
        expected = metric_values(json.loads((reference / 'metrics.json').read_text()))
        metrics = {}
        for key, value in measured.items():
            target = expected[key]
            matches = value is target if value is None or target is None else math.isclose(
                value, target, rel_tol=0, abs_tol=report['metric_atol'])
            metrics[key] = {'actual': value, 'reference': target, 'matches': matches,
                            'difference': None if value is None or target is None else value - target}
        predictions = compare(pd.read_parquet(actual / 'predictions.parquet'),
                              pd.read_parquet(reference / 'predictions.parquet'), 'score')
        passed = all(item['matches'] for item in metrics.values()) and predictions['status'] == 'PASS'
        report['methods'][method] = {'status': 'PASS' if passed else 'MISMATCH',
                                     'metrics': metrics, 'predictions': predictions}
        if not passed:
            report['status'] = 'MISMATCH'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run_root.resolve())
    destination = args.run_root / 'reference-comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['status']}: {destination}")
    if report['status'] != 'PASS':
        raise SystemExit('The generated results differ from the reference; inspect reference-comparison.json.')


if __name__ == '__main__':
    main()
