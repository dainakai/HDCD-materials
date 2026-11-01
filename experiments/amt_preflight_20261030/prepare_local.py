"""Install the frozen experiment engine and frame manifests in the audit folder."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO.parent/'audit/revision-preflight-20261030'
ARCHIVE=Path('/media/dai-server/DATAM2/siderepos/260804/paper-range-only-20260930')

def same_or_copy(source,target):
    if target.exists():
        if source.read_bytes()!=target.read_bytes():
            raise ValueError(f'Preserving modified file; inputs differ: {target}')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)

for name in ['config','data','models','results','figures','logs','tmp','cache']:
    (ROOT/name).mkdir(parents=True,exist_ok=True)
manifest={}
for source in sorted((HERE/'frozen_engine').rglob('*')):
    if source.is_file() and '__pycache__' not in source.parts:
        relative=source.relative_to(HERE/'frozen_engine')
        same_or_copy(source,ROOT/'engine'/relative)
        manifest[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
same_or_copy(HERE/'protocol.json',ROOT/'config/protocol.json')
for seed in range(260740,260746):
    same_or_copy(ARCHIVE/f'config/seed{seed}-frames.csv',ROOT/f'config/seed{seed}-frames.csv')
(ROOT/'config/engine-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Prepared {len(manifest)} frozen engine files in {ROOT}')
