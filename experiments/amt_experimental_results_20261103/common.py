"""Independent post-review analysis; never mutates the campaign or its annotations."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
CAMPAIGN = WORK / 'audit/revision-experimental-20261102'
ROOT = WORK / 'audit/revision-experimental-results-20261103'
PRIVATE = ROOT / 'snapshot'
PACKAGE = ROOT / 'package'
DATA = PACKAGE / 'data'
FIGURES = PACKAGE / 'figures'
SCENES = [12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
MODEL_SHA = 'd648ebf407d18ff66b54670711dbebfe8997b1fa6a502a982b39803a19fda114'
FRAMES = 10917
FPS = 4000.
VOLUME_CM3 = 10.24 * 10.24 * 102.3 / 1000
EXPOSURE = VOLUME_CM3 * 1e-6 * FRAMES / FPS

def stamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
