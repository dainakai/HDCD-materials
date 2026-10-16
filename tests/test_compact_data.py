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


def make_pair(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    parent = "data/candidates/validation/seed260745/wide.parquet"
    child = "data/features/validation/seed260745/static77.parquet"
    n = 4000
    a = np.random.default_rng(71).normal(size=n)
    frame = pa.table(
        {
            "vanished_track_id": np.arange(n),
            "survivor_track_id": np.arange(n) + 10000,
            "candidate_frame": np.arange(n) % 13,
            "x": a,
            "large_integer": np.arange(n, dtype=np.int64) + 2**60,
            "text": pa.array(["µm", None] * (n // 2)),
        }
    )
    selected = frame.take(pa.array(np.arange(n - 1, -1, -1)))
    delta = np.asarray(selected["x"]).copy()
    delta[2] = np.nextafter(delta[2], np.inf)
    delta[3] = -0.0
    selected = selected.set_column(selected.schema.get_field_index("x"), "x", pa.array(delta))
    selected = selected.append_column("nullable", pa.array([None, 1.0] * (n // 2)))
    for name, table in [(parent, frame), (child, selected)]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
    builder = compact.Builder(source, target)
    for name in [parent, child]:
        p = source / name
        builder.build(name, {"bytes": p.stat().st_size, "sha256": compact.file_hash(p)})
    manifest = {"files": builder.recipes, "objects": builder.store.objects}
    return source, target, parent, child, manifest


def test_reordered_rows_low_bits_nulls_and_large_integers_roundtrip(tmp_path):
    source, target, parent, child, manifest = make_pair(tmp_path)
    assert manifest["files"][child]["parents"]
    reader = compact.Reader(target, manifest)
    completed = {}
    reader.materialize(child, completed)
    for name in [parent, child]:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert len(completed) == 2
    again = {}
    compact.Reader(target, manifest).materialize(child, again)
    assert again == {child: "already_verified"}


def test_existing_modified_data_are_not_overwritten(tmp_path):
    _, target, _, child, manifest = make_pair(tmp_path)
    path = target / child
    path.parent.mkdir(parents=True)
    path.write_bytes(b"user changes")
    with pytest.raises(ValueError, match="Existing file differs"):
        compact.Reader(target, manifest).materialize(child, {})
    assert path.read_bytes() == b"user changes"


def test_corrupt_payload_is_detected_before_installing_output(tmp_path):
    _, target, parent, _, manifest = make_pair(tmp_path)
    first = manifest["files"][parent]["columns"][0]["object"]
    chunk = manifest["objects"][first]["parts"][0]["path"]
    (target / "data_sources" / chunk).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        compact.Reader(target, manifest).materialize(parent, {})
    assert not (target / parent).exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute/file"])
def test_path_escape_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe relative path"):
        compact.contained(tmp_path, name)


def test_file_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Unsafe relative path"):
        compact.contained(root, "escape/value")


def test_nonstandard_parquet_layout_uses_exact_binary_recipe(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    p = source / "small.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4]}), p, row_group_size=2)
    b = compact.Builder(source, target)
    b.build(p.name, {"bytes": p.stat().st_size, "sha256": compact.file_hash(p)})
    assert b.recipes[p.name]["kind"] == "binary"
    compact.Reader(target, {"files": b.recipes, "objects": b.store.objects}).materialize(p.name, {})
    assert (target / p.name).read_bytes() == p.read_bytes()


def test_filesystem_without_hard_links_preserves_exact_output(tmp_path, monkeypatch):
    source, target, parent, _, manifest = make_pair(tmp_path)

    def unsupported(*args, **kwargs):
        raise OSError(compact.errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(compact.os, "link", unsupported)
    compact.Reader(target, manifest).materialize(parent, {})
    assert (target / parent).read_bytes() == (source / parent).read_bytes()
