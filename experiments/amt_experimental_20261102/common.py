"""Experimental inference reuses the verified numerical engine and fixed weights."""
from pathlib import Path
import sys
import os
import json
import hashlib
import datetime as dt

SCRIPT = Path(__file__).resolve().parent
REPO = SCRIPT.parents[1]
WORKSPACE = REPO.parent
NUMERICAL = WORKSPACE / 'audit/revision-full-20261101'
ROOT = WORKSPACE / 'audit/revision-experimental-20261102'
SOURCE = NUMERICAL / 'upstream/production'
RAW = Path('/media/dai-server/HDD16TB/repos/260703/260715')
PREPARATION = WORKSPACE / 'audit/revision-review-20261101/experimental-preparation'
GPU_PYTHON = Path('/media/dai-server/M2Disk2TB/repos/260712/.venv/bin/python')
CPU_PYTHON = REPO / '.venv/bin/python'
SCENES = [12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
FRAMES = 10917
RUN_ID = 'amt-102p3mm-phase3-experimental-20261102'
MODEL = NUMERICAL / 'models/ours'
sys.dont_write_bytecode = True
sys.path.insert(0, str(NUMERICAL / 'src'))
sys.path.append(str(REPO / '.venv/lib/python3.12/site-packages'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONPATH'] = str(NUMERICAL / 'src')
for name in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'NUMBA_CACHE_DIR', 'YOLO_CONFIG_DIR', 'TMPDIR']:
    directory = ROOT / ('tmp' if name == 'TMPDIR' else 'cache') / name.lower()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '4'
os.environ['JULIA_DEPOT_PATH'] = str(ROOT / 'cache/julia') + ':' + str(NUMERICAL / 'cache/julia') + ':/home/dai-server/.julia'
os.environ['JULIA_PKG_OFFLINE'] = 'true'
os.environ['JULIA_NUM_THREADS'] = '4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()


def emit(**values):
    print(json.dumps({'time': stamp(), **values}, ensure_ascii=False), flush=True)


def scene(n):
    return ROOT / f'scenes/scene{n:04d}'


def verify_engine():
    manifest = read(REPO / 'experiments/amt_full_20261101/engine-manifest.json')['files']
    for name, expected in manifest.items():
        assert sha(NUMERICAL / 'src' / name) == expected, name
    freeze = read(PREPARATION / 'model-freeze.json')
    assert sha(MODEL / 'model.json') == freeze['models']['ours']['model_json_sha256']
    for name, expected in read(MODEL / 'model.json')['files'].items():
        assert sha(MODEL / name) == expected, name
    assert read(NUMERICAL / 'finalized.json')['status'] == 'PASS'
    return freeze
