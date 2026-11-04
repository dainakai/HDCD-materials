from bootstrap import ROOT, REPO, SCRIPT
from pathlib import Path
import datetime as dt
import json
import numpy as np
import pandas as pd
from ra3d.io import sha256, write_json, write_table, read_table

GENERATION = ROOT / 'generation'
import os
import sys
GPU_PYTHON = Path(os.environ.get('HDCD_GPU_PYTHON', sys.executable))
CPU_PYTHON = Path(sys.executable)
JULIA = os.environ.get('HDCD_JULIA', 'julia')
SOURCE = REPO / 'reference/measurement'
TRAIN = list(range(260740, 260745))
SEEDS = TRAIN + [260745]


def emit(**values):
    print(json.dumps({'time': dt.datetime.now().astimezone().isoformat(), **values}), flush=True)


def split(seed):
    return 'validation' if seed == 260745 else 'train'


def path(seed, method, name):
    branch = {'ours': 'features', 'depth_free': 'baseline2d', 'minip2d': 'minip2d'}[method]
    p = ROOT / f'data/{branch}/{split(seed)}/seed{seed}/{name}'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def tracking(seed, method):
    return ROOT / f'runs/{"minip-tracks" if method == "minip2d" else "tracking"}/seed{seed}'


def measurement(seed):
    return ROOT / f'runs/measurement/seed{seed}'


def truth_path(seed, filename='collision_events.parquet'):
    return ROOT / f'data/labels/{split(seed)}/seed{seed}/{filename}'


def events(seed):
    return read_table(truth_path(seed))


def wide_path(seed, method):
    return path(seed, method, 'wide.parquet')


def minip_dir(seed):
    return ROOT / f'runs/minip/seed{seed}'


def labels(frame, seed, method):
    from ra3d.strict_labels import attach_identity_labels
    mapping = read_table(tracking(seed, method) / 'mapping.parquet')
    out, audit, _ = attach_identity_labels(frame, events(seed), mapping,
                                         z_tolerance_um=15000. if method == 'ours' else None)
    out['synth_seed'] = seed
    return out, audit
