"""Paths and isolated runtime for the AMT revision's controlled experiments."""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ROOT = WORKSPACE / 'audit/revision-preflight-20261030'
ARCHIVE = Path('/media/dai-server/DATAM2/siderepos/260804/paper-range-only-20260930')
TRUTH = ARCHIVE.parent / 'zenodo-collision-reproducibility'
TRAIN = list(range(260740, 260745))
SEEDS = TRAIN + [260745]
sys.path.insert(0, str(ROOT / 'engine'))
sys.path.append(str(REPO / '.venv/lib/python3.12/site-packages'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TMPDIR']:
    destination = ROOT / ('tmp' if name == 'TMPDIR' else 'cache') / name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(destination)
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '4')

import datetime
import json
import numpy as np
import pandas as pd
from ra3d.io import read_table, write_table, write_json, sha256


def emit(**values):
    print(json.dumps({'time': datetime.datetime.now().astimezone().isoformat(), **values}), flush=True)


def split(seed):
    return 'validation' if seed == 260745 else 'train'


def feature_path(seed, name):
    return ARCHIVE / f'data/features/{split(seed)}/seed{seed}/{name}'


def tracks_path(seed, name):
    return ARCHIVE / f'data/tracking/{split(seed)}/seed{seed}/{name}'


def truth_path(seed, name='collision_events.parquet'):
    return TRUTH / f'data/labels/{split(seed)}/seed{seed}/{name}'


def data_path(seed, name):
    p = ROOT / f'data/seed{seed}/{name}'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def summarize(m):
    result = {k: m[k] for k in ['event_ap', 'maximum_f1', 'maximum_recall', 'predicted_events']}
    for target in [.8, .9]:
        point = m[f'operating_at_recall_{target:.2f}']
        result[f'fp_r{target}'] = None if point is None else int(point['false_positive_events'])
        result[f'precision_r{target}'] = None if point is None else point['precision']
    return result
