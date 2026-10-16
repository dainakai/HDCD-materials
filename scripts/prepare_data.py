"""Prepare the test evaluation and paper figures from reduced numerical inputs.

Trajectory features are calculated locally; the reference serialization is then
recovered using recorded bitwise differences. Recipes are data, not executable code.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHUNK_BYTES = 24 * 1024**2
VERSIONS = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "pyarrow": "25.0.1",
    "zstandard": "0.25.0",
}
KEYS = ["vanished_track_id", "survivor_track_id", "candidate_frame"]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    target = (root / p).resolve()
    if p.is_absolute() or ".." in p.parts or not target.is_relative_to(root):
        raise ValueError(f"Unsafe relative path: {relative}")
    return target


def check_versions():
    actual = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pa.__version__,
        "zstandard": zstd.__version__,
    }
    if actual != VERSIONS:
        raise ValueError(
            f"Exact file reconstruction requires {VERSIONS}; found {actual}. "
            "Install requirements-core-reference.txt."
        )
    return actual


def numeric(array):
    return pa.types.is_integer(array.type) or pa.types.is_floating(array.type)


def numeric_bytes(array):
    values = np.asarray(pc.fill_null(array, 0)).astype(array.type.to_pandas_dtype(), copy=True)
    values = values.astype(values.dtype.newbyteorder("<"), copy=False)
    valid = np.asarray(pc.is_valid(array))
    mask = None if valid.all() else np.packbits(valid, bitorder="little").tobytes()
    return values.tobytes(), mask


def ipc_bytes(array):
    out = pa.BufferOutputStream()
    with pa.ipc.new_stream(out, pa.schema([("value", array.type)])) as writer:
        writer.write_batch(pa.record_batch([array], names=["value"]))
    return out.getvalue().to_pybytes()


class Store:
    def __init__(self, root, objects=None):
        self.root = Path(root)
        self.objects = objects if objects is not None else {}
        self.compressor = zstd.ZstdCompressor(level=9)

    def put(self, raw, encoded=None):
        key = digest(raw)
        if key in self.objects:
            return key
        packed = self.compressor.compress(raw) if encoded is None else encoded
        parts = []
        for i, start in enumerate(range(0, len(packed), CHUNK_BYTES)):
            part = packed[start : start + CHUNK_BYTES]
            name = f"objects/{key[:2]}/{key}.{i:03d}.zstpart"
            path = contained(self.root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(part)
            parts.append({"path": name, "bytes": len(part), "sha256": digest(part)})
        self.objects[key] = {"bytes": len(raw), "parts": parts}
        return key

    def get(self, key):
        item = self.objects[key]
        packed = bytearray()
        for part in item["parts"]:
            data = contained(self.root, part["path"]).read_bytes()
            if len(data) != part["bytes"] or digest(data) != part["sha256"]:
                raise ValueError(f"Compact input checksum mismatch: {part['path']}")
            packed.extend(data)
        raw = zstd.ZstdDecompressor().decompress(packed, max_output_size=max(item["bytes"], 1))
        if len(raw) != item["bytes"] or digest(raw) != key:
            raise ValueError(f"Decoded object checksum mismatch: {key}")
        return raw


def parent_candidates(name):
    p = Path(name)
    if name.startswith("data/features/"):
        scene = "/".join(p.parts[2:4])
        stem = f"data/features/{scene}"
        return {
            "stage1_training.parquet": [f"data/candidates/{scene}/wide.parquet"],
            "static77.parquet": [f"data/candidates/{scene}/wide.parquet"],
            "static_scored.parquet": [f"{stem}/static77.parquet"],
            "paper92.parquet": [f"{stem}/static_scored.parquet", f"{stem}/static77.parquet"],
            "image_complete92.parquet": [f"{stem}/paper92.parquet"],
        }.get(p.name, [])
    if name.startswith("data/baseline2d/"):
        scene = "/".join(p.parts[2:4])
        stem = f"data/baseline2d/{scene}"
        return {
            "wide.parquet": [f"data/candidates/{scene}/wide.parquet"],
            "static.parquet": [f"{stem}/wide.parquet", f"data/features/{scene}/static77.parquet"],
            "gate.parquet": [f"{stem}/static.parquet", f"{stem}/wide.parquet"],
            "full.parquet": [f"{stem}/static.parquet"],
        }.get(p.name, [])
    if name.startswith("data/minip2d/"):
        stem = str(p.parent)
        return {
            "gate.parquet": [f"{stem}/static.parquet"],
            "full.parquet": [f"{stem}/static.parquet"],
            "smoothedtrajs.parquet": [f"{stem}/trajectory.parquet"],
        }.get(p.name, [])
    if name.startswith("data/detections/") and p.name == "raw_detections.parquet":
        return [str(p.with_name("particles_3d.parquet"))]
    if name.startswith("data/tracking/") and p.name == "smoothedtrajs.parquet":
        return [str(p.with_name("trajectory3d.parquet"))]
    if "/experimental/" in name and "/features/" in name:
        stem = str(p.parent)
        return {
            "stage1_shortlist.parquet": [f"{stem}/permissive_endpoint.parquet"],
            "static_enriched.parquet": [f"{stem}/stage1_shortlist.parquet"],
            "static_scored.parquet": [f"{stem}/static_enriched.parquet"],
            "image_enriched.parquet": [f"{stem}/static_scored.parquet"],
            "full_features.parquet": [f"{stem}/static_scored.parquet", f"{stem}/image_enriched.parquet"],
        }.get(p.name, [])
    return []


def parquet_bytes(table, compression):
    sink = pa.BufferOutputStream()
    pq.write_table(table.combine_chunks(), sink, compression=compression)
    return sink.getvalue().to_pybytes()


class Builder:
    """Maintainer-side assembly from the protected full reference directory."""

    def __init__(self, source, destination):
        self.source = Path(source)
        self.destination = Path(destination)
        self.store = Store(self.destination / "data_sources")
        self.recipes = {}
        self.tables = OrderedDict()

    def table(self, name):
        if name not in self.tables:
            self.tables[name] = pq.read_table(self.source / name).combine_chunks()
            while len(self.tables) > 6:
                self.tables.popitem(last=False)
        self.tables.move_to_end(name)
        return self.tables[name]

    def build(self, name, expected):
        if name in self.recipes:
            return
        path = self.source / name
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Full reference changed: {name}")
        row = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
        if name.startswith("data/labels/") and path.name == "particles.parquet":
            schema = pq.read_schema(path)
            row.update(
                kind="simulation",
                seed=int(path.parent.name.removeprefix("seed")),
                schema=base64.b64encode(schema.serialize()).decode(),
                compression=pq.read_metadata(path).row_group(0).column(0).compression.lower(),
            )
            self.recipes[name] = row
            return
        if path.suffix != ".parquet":
            row.update(kind="binary", chunks=[])
            with path.open("rb") as stream:
                for raw in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    row["chunks"].append(self.store.put(raw))
            self.recipes[name] = row
            return
        table = self.table(name)
        codec = pq.read_metadata(path).row_group(0).column(0).compression.lower()
        # Combining Arrow chunks is required to reproduce original page boundaries.
        if digest(parquet_bytes(table, codec)) != expected["sha256"]:
            row.update(kind="binary", chunks=[])
            with path.open("rb") as stream:
                for raw in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    row["chunks"].append(self.store.put(raw))
            self.recipes[name] = row
            return
        row.update(
            kind="parquet",
            rows=table.num_rows,
            compression=codec,
            schema=base64.b64encode(table.schema.serialize()).decode(),
            columns=[],
            parents={},
        )
        mappings = {}
        import pandas as pd

        for parent in parent_candidates(name):
            if parent not in self.recipes:
                continue
            pt = self.table(parent)
            common = KEYS if all(k in table.column_names and k in pt.column_names for k in KEYS) else None
            if common is None and all(
                k in table.column_names and k in pt.column_names for k in ["track_id", "frame"]
            ):
                common = ["track_id", "frame"]
            if common is None:
                if table.num_rows != pt.num_rows:
                    continue
                # Only explicit detection parent pairs use positional alignment.
                if not name.startswith("data/detections/"):
                    continue
                loc = np.arange(table.num_rows, dtype=np.int64)
            else:
                pi = pd.MultiIndex.from_frame(pt.select(common).to_pandas())
                if not pi.is_unique:
                    continue
                ti = pd.MultiIndex.from_frame(table.select(common).to_pandas())
                loc = pi.get_indexer(ti)
            mappings[parent] = loc
        for field, column in zip(table.schema, table.columns, strict=True):
            array = column.combine_chunks()
            if not numeric(array):
                row["columns"].append({"kind": "ipc", "object": self.store.put(ipc_bytes(array))})
                continue
            raw, mask = numeric_bytes(array)
            best_raw = raw
            best_compressed = self.store.compressor.compress(raw)
            best_parent = None
            for parent, loc in mappings.items():
                pt = self.table(parent)
                if field.name not in pt.column_names or pt[field.name].type != field.type:
                    continue
                base = pt[field.name].take(pa.array(np.maximum(loc, 0))).combine_chunks()
                base_raw, _ = numeric_bytes(base)
                base_values = np.frombuffer(base_raw, dtype=f"<u{field.type.bit_width // 8}").copy()
                base_values[loc < 0] = 0
                delta = np.bitwise_xor(np.frombuffer(raw, dtype=base_values.dtype), base_values).tobytes()
                packed = self.store.compressor.compress(delta)
                if len(packed) < len(best_compressed):
                    best_raw, best_compressed, best_parent = delta, packed, parent
            spec = {
                "kind": "numeric",
                "object": self.store.put(best_raw, best_compressed),
                "validity": self.store.put(mask) if mask is not None else None,
            }
            if best_parent is not None:
                spec["parent"] = best_parent
                if best_parent not in row["parents"]:
                    loc = mappings[best_parent]
                    row["parents"][best_parent] = self.store.put(loc.astype("<i4").tobytes())
            row["columns"].append(spec)
        self.recipes[name] = row


class Reader:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.manifest = manifest
        self.store = Store(self.root / "data_sources", manifest["objects"])
        self.tables = OrderedDict()
        self.active = set()

    def table(self, name):
        if name not in self.tables:
            path = contained(self.root, name)
            if file_hash(path) != self.manifest["files"][name]["sha256"]:
                raise ValueError(f"Dependency checksum mismatch: {name}")
            self.tables[name] = pq.read_table(path).combine_chunks()
            while len(self.tables) > 6:
                self.tables.popitem(last=False)
        self.tables.move_to_end(name)
        return self.tables[name]

    def materialize(self, name, completed):
        if name in completed:
            return
        if name in self.active:
            raise ValueError(f"Cyclic recipe dependency: {name}")
        self.active.add(name)
        row = self.manifest["files"][name]
        target = contained(self.root, name)
        if target.exists():
            if not target.is_file() or file_hash(target) != row["sha256"]:
                raise ValueError(f"Existing file differs; move it aside before preparing data: {target}")
            completed[name] = "already_verified"
            self.active.remove(name)
            return
        for parent in row.get("parents", {}):
            self.materialize(parent, completed)
        for dependency in row.get("requires", []):
            self.materialize(dependency, completed)
        if row["kind"] == "parquet":
            schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(row["schema"])))
            if len(schema) != len(row["columns"]):
                raise ValueError("Column recipe length differs from schema")
            mappings = {
                p: np.frombuffer(self.store.get(obj), dtype="<i4") for p, obj in row["parents"].items()
            }
            arrays = []
            for field, spec in zip(schema, row["columns"], strict=True):
                raw = self.store.get(spec["object"])
                if spec["kind"] == "ipc":
                    array = pa.ipc.open_stream(raw).read_all().column(0).combine_chunks()
                elif spec["kind"] == "numeric":
                    if "parent" in spec:
                        parent = spec["parent"]
                        loc = mappings[parent]
                        if len(loc) != row["rows"] or np.any(loc < -1):
                            raise ValueError("Invalid parent row mapping")
                        base = (
                            self.table(parent)[field.name].take(pa.array(np.maximum(loc, 0))).combine_chunks()
                        )
                        base_raw, _ = numeric_bytes(base)
                        values = np.frombuffer(base_raw, dtype=f"<u{field.type.bit_width // 8}").copy()
                        values[loc < 0] = 0
                        raw = np.bitwise_xor(values, np.frombuffer(raw, dtype=values.dtype)).tobytes()
                    mask = self.store.get(spec["validity"]) if spec["validity"] is not None else None
                    array = pa.Array.from_buffers(
                        field.type,
                        row["rows"],
                        [pa.py_buffer(mask) if mask is not None else None, pa.py_buffer(raw)],
                    )
                else:
                    raise ValueError(f"Unknown column recipe: {spec['kind']}")
                arrays.append(array)
            table = pa.Table.from_arrays(arrays, schema=schema)
            content = parquet_bytes(table, row["compression"])
        elif row["kind"] == "trajectory_features":
            from ra3d.features import static_features
            from ra3d.io import legacy_candidate_csv
            from ra3d.tracks import read_tracks, materialize_tracks

            candidates = self.table(row["candidates"]).to_pandas()
            tracks = read_tracks(contained(self.root, row["tracks"]))
            workspace = self.root / "runs"
            workspace.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".trajectory-features-", dir=workspace) as work:
                track_dir = materialize_tracks(tracks, Path(work) / "tracks")

                def progress(state):
                    if state["rows"] % 10000 == 0:
                        print(f"Trajectory features: {state['rows']}/{state['total']}", flush=True)

                frame = static_features(
                    legacy_candidate_csv(candidates), tracks, track_dir, progress=progress
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                content = parquet_bytes(table, "zstd")
        elif row["kind"] == "simulation":
            import pandas as pd

            workspace = self.root / "runs"
            workspace.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".collision-labels-", dir=workspace) as work:
                subprocess.run(
                    [
                        sys.executable,
                        str(self.root / "scripts/generate_holograms.py"),
                        "--seed",
                        str(row["seed"]),
                        "--stage",
                        "labels",
                        "--output",
                        work,
                    ],
                    cwd=self.root,
                    check=True,
                )
                generated = Path(work) / f"seed{row['seed']}/labels/particles.csv"
                frame = pd.read_csv(generated, float_precision="round_trip", low_memory=False)
                schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(row["schema"])))
                table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                table = table.replace_schema_metadata(schema.metadata)
                content = parquet_bytes(table, row["compression"])
        elif row["kind"] == "binary":
            content = b"".join(self.store.get(key) for key in row["chunks"])
        else:
            raise ValueError(f"Unknown file recipe: {row['kind']}")
        if len(content) != row["bytes"] or digest(content) != row["sha256"]:
            raise ValueError(f"Reconstructed file differs from the reference: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".prepare-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            # Hard-link installation is atomic and never overwrites an existing file.
            try:
                os.link(temporary, target)
            except OSError as error:
                if error.errno not in {errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV}:
                    raise
                # FAT/exFAT do not support hard links. Exclusive creation still
                # protects user files; an interrupted partial output is detected
                # by its checksum on the next invocation.
                with target.open("xb") as stream:
                    stream.write(content)
        finally:
            temporary.unlink()
        completed[name] = "reconstructed_sha256_exact"
        self.active.remove(name)
        print(f"Verified {name}", flush=True)


def profile_paths(files, profile):
    if profile in {"all", "paper"}:
        return list(files)
    if profile == "evaluation":
        return [
            n
            for n in files
            if not n.startswith("data/paper_figures/") and not n.endswith("/particles.parquet")
        ]
    if profile == "figures":
        return [
            n
            for n in files
            if n.startswith(("data/paper_figures/", "data/predictions/")) or n.endswith("/particles.parquet")
        ]
    raise ValueError(f"Unknown profile: {profile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--profile",
        choices=["all", "paper", "evaluation", "figures"],
        default="all",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="List selected outputs without reconstructing")
    parser.add_argument("--output", type=Path, help="Write a JSON execution report")
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root.resolve()
    source = root / "data_sources/recipes.json"
    expected = (root / "data_sources/recipes.sha256").read_text().split()[0]
    if file_hash(source) != expected:
        raise ValueError("Recipe manifest checksum mismatch")
    manifest = json.loads(source.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported compact data format")
    selected = profile_paths(manifest["files"], args.profile)
    if args.list:
        for name in selected:
            print(name)
        return
    versions = check_versions()
    required = set()

    def visit(name):
        if name in required:
            return
        required.add(name)
        for parent in manifest["files"][name].get("parents", {}):
            visit(parent)
        for dependency in manifest["files"][name].get("requires", []):
            visit(dependency)

    for name in selected:
        visit(name)
    if not args.verify_only:
        missing_bytes = sum(
            manifest["files"][name]["bytes"] for name in required if not contained(root, name).exists()
        )
        if shutil.disk_usage(root).free < missing_bytes + 2 * 1024**3:
            raise ValueError(
                f"Need approximately {missing_bytes / 1e9:.2f} GB of additional output space "
                "plus 2 GiB working space. Choose a filesystem with sufficient free space."
            )
    completed = {}
    reader = Reader(root, manifest)
    for name in selected:
        if args.verify_only:
            path = contained(root, name)
            if not path.is_file() or file_hash(path) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Missing or changed prepared file: {name}")
            completed[name] = "verified"
        else:
            reader.materialize(name, completed)
    report = {
        "status": "PASS",
        "profile": args.profile,
        "file_count": len(completed),
        "exact_expected_sha256": True,
        "versions": versions,
        "elapsed_seconds": time.monotonic() - started,
        "files": completed,
    }
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(root) and not output.is_relative_to(root / "runs"):
            parser.error("Write reports under runs/ or outside this repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
