#!/usr/bin/env python3
"""Recalculate Tables 4 and 5 from the supplied models and complete test features."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose an empty output directory')
    os.environ['HDCD_RUN_ROOT'] = str(a.output.resolve())
    sys.path.insert(0, str(REPO / 'src'))
    sys.path.insert(0, str(REPO / 'scripts/benchmark'))
    import pandas as pd
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from reproduce import terminal
    from evaluate import stages, summary_row
    from ra3d.models import load_model
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        report = terminal(a.output / 'predictions', 'cpu')
        table4 = pd.DataFrame([{'method': name, **summary_row(row['metrics'])} for name, row in report.items()])
        table4.to_csv(a.output / 'table4.csv', index=False)
        truth = pd.read_parquet(REPO / 'data/test/collision_events.parquet')
        table5 = pd.DataFrame(stages(load_model(REPO / 'data/models/ours'), truth,
                                    REPO / 'data/test/trajectory.parquet', REPO / 'data/test/ours.parquet'))
        reference = pd.read_csv(REPO / 'reference/evaluation/table5.csv')
        for col in reference.select_dtypes('number'):
            np.testing.assert_allclose(table5[col], reference[col], rtol=0, atol=1e-10)
        table5.to_csv(a.output / 'table5.csv', index=False)
        for number, table in [(4, table4), (5, table5)]:
            (a.output / f'table{number}.tex').write_text(table.to_latex(index=False, float_format=lambda v: f'{v:.6f}'))
        (a.output / 'verification.json').write_text(json.dumps({'status': 'PASS', 'tables': [4, 5]}, indent=2) + '\n')
        print('PASS: Tables 4 and 5 agree with the reference evaluation.')


if __name__ == '__main__':
    main()
