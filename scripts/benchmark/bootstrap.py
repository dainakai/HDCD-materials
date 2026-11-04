"""Shared paths for a self-contained numerical benchmark run."""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('HDCD_RUN_ROOT', REPO / 'runs/benchmark')).resolve()
SCRIPT = Path(__file__).resolve().parent
if ROOT == REPO or any(ROOT == REPO / name or REPO / name in ROOT.parents
                       for name in ['src', 'data', 'data_sources', 'reference', 'scripts']):
    raise ValueError('Choose a run directory outside the distributed input directories')
sys.path.insert(0, str(REPO / 'src'))
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONPATH'] = str(REPO / 'src')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH',
             'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'NUMBA_CACHE_DIR', 'YOLO_CONFIG_DIR']:
    folder = ROOT / 'cache' / name.lower()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(folder))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JULIA_NUM_THREADS']:
    os.environ.setdefault(name, '4')
for name in ['config', 'logs', 'jobs', 'results', 'figures', 'models']:
    (ROOT / name).mkdir(parents=True, exist_ok=True)
