"""Apply an optional allocator cap before running an unchanged stage entrypoint."""
import runpy
import sys
import torch

fraction=float(sys.argv[1])
target=sys.argv[2]
assert 0 < fraction <= 1
torch.cuda.set_per_process_memory_fraction(fraction,0)
sys.argv=sys.argv[2:]
sys.path.insert(0,str(__import__('pathlib').Path(target).resolve().parent))
runpy.run_path(target,run_name='__main__')
