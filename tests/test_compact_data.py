"""Losslessness, interruption safety, and untrusted-path checks for preparation."""

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_data.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name("compact_data.py")
spec = importlib.util.spec_from_file_location("compact", SCRIPT)
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


def make_payload(root):
    import json
    import zstandard
    source = root / 'data_sources'
    source.mkdir()
    values = pa.table({'large': [2**60, 2**60+1], 'float': [np.nextafter(1., 2.), -0.],
                       'text': ['µm', None]})
    temporary = root / 'source.parquet'
    pq.write_table(values, temporary, row_group_size=1)
    original = temporary.read_bytes()
    compressed = zstandard.ZstdCompressor().compress(original)
    parts = []
    for i, payload in enumerate([compressed[:31], compressed[31:97], compressed[97:]]):
        path = source / f'part{i}'
        path.write_bytes(payload)
        parts.append({'path': str(path.relative_to(root)), 'sha256': compact.digest(path)})
    item = {'profile': 'test', 'destination': 'data/test/input.parquet',
            'sha256': compact.digest(temporary), 'parts': parts}
    (source / 'manifest.json').write_text(json.dumps({'objects': [item]}))
    return original, item


def test_chunked_stream_preserves_parquet_bytes_and_resumes(tmp_path):
    original, item = make_payload(tmp_path)
    compact.prepare('test', tmp_path)
    assert (tmp_path / item['destination']).read_bytes() == original
    assert compact.prepare('test', tmp_path)['status'] == 'PASS'


def test_existing_modified_data_are_not_overwritten(tmp_path):
    _, item = make_payload(tmp_path)
    target = tmp_path / item['destination']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'user changes')
    with pytest.raises(ValueError, match='Existing file differs'):
        compact.prepare('test', tmp_path)
    assert target.read_bytes() == b'user changes'


def test_corrupt_payload_is_detected_before_installing_output(tmp_path):
    _, item = make_payload(tmp_path)
    (tmp_path / item['parts'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        compact.prepare('test', tmp_path)
    assert not (tmp_path / item['destination']).exists()


def test_wrong_decoded_checksum_leaves_no_partial_file(tmp_path):
    import json
    _, item = make_payload(tmp_path)
    item['sha256'] = '0' * 64
    (tmp_path / 'data_sources/manifest.json').write_text(json.dumps({'objects': [item]}))
    with pytest.raises(ValueError, match='Decoded checksum mismatch'):
        compact.prepare('test', tmp_path)
    assert list((tmp_path / 'data/test').iterdir()) == []


@pytest.mark.parametrize('name', ['../outside', '/absolute/file'])
def test_path_escape_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='Unsafe relative path'):
        compact.contained(tmp_path, name)


def test_file_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'escape').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='Unsafe relative path'):
        compact.contained(root, 'escape/value')
