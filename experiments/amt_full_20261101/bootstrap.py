"""Isolated full-revision runtime; external programs and weights remain read-only."""
from pathlib import Path
import os
import sys
import hashlib
import json
import shutil

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent / 'audit/revision-full-20261101'
SCRIPT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for relative, expected in json.loads((SCRIPT / 'engine-manifest.json').read_text())['files'].items():
    source = SCRIPT / 'frozen_engine' / relative
    destination = ROOT / 'src' / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Runtime source differs from its frozen manifest: {destination}')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.append(str(REPO / '.venv/lib/python3.12/site-packages'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONPATH'] = str(ROOT / 'src')
for name in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'NUMBA_CACHE_DIR', 'YOLO_CONFIG_DIR', 'TMPDIR']:
    path = ROOT / ('tmp' if name == 'TMPDIR' else 'cache') / name.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '4')
os.environ['JULIA_DEPOT_PATH'] = str(ROOT / 'cache/julia') + ':/home/dai-server/.julia'
os.environ['JULIA_PKG_OFFLINE'] = 'true'
os.environ.setdefault('JULIA_NUM_THREADS', '4')
