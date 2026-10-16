#!/usr/bin/env python3
"""Reduce original particle tables to full-recording diameter counts for Figures 14–15.

The source CSV has columns raw_scene_num and path. Paths may be relative to that
CSV. Each particle table needs frame and final_diameter_um (or diameter_um).
The published figure workflow starts from the resulting compact count table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from figure_support.scene_index import SCENE_NUMBERS, paper_scene_index

RECORDED_FRAMES = 10917
HISTORICAL_FRAMES = 10904
EDGES = np.arange(25.0, 301.0, 5.0)


def count_particles(path: Path):
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
    else:
        schema = pl.scan_csv(path).collect_schema()
    diameter = next((name for name in ["final_diameter_um", "diameter_um", "slice_diam_pred_um"] if name in schema), None)
    if "frame" not in schema or diameter is None:
        raise ValueError(f"Missing frame or measured diameter: {path}")
    columns = ["frame", diameter]
    if path.suffix == ".parquet":
        values = pl.read_parquet(path, columns=columns)
    else:
        values = pl.read_csv(path, columns=columns, schema_overrides={name: pl.Float64 for name in columns}, n_threads=4)
    frame, size = (values[name].to_numpy() for name in columns)
    finite = np.isfinite(frame) & np.isfinite(size) & (size > 0)
    frame, size = frame[finite], size[finite]
    if not len(frame) or np.any(frame != np.floor(frame)) or frame.min() < 0 or frame.max() >= RECORDED_FRAMES:
        raise ValueError(f"Invalid recorded frame range: {path}")
    supported = frame < HISTORICAL_FRAMES
    early = np.histogram(size[supported], bins=EDGES)[0]
    tail = np.histogram(size[~supported], bins=EDGES)[0]
    # NumPy includes the final right edge, matching the saved analysis.
    return early, tail, {"diameter_column": diameter, "finite_positive_rows": len(frame),
                         "last_13_frame_rows": int((~supported).sum()),
                         "frame_min": int(frame.min()), "frame_max": int(frame.max())}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        parser.error("Choose new output and audit paths")
    sources = pd.read_csv(args.sources)
    if sources.raw_scene_num.duplicated().any() or set(sources.raw_scene_num) != set(SCENE_NUMBERS):
        parser.error("Provide exactly one particle table for each of the 14 paper scenes")
    rows, audits = [], []
    for source in sources.sort_values("raw_scene_num").itertuples():
        path = Path(source.path)
        if not path.is_absolute():
            path = args.sources.parent / path
        early, tail, audit = count_particles(path)
        for i, (low, high) in enumerate(zip(EDGES[:-1], EDGES[1:], strict=True)):
            rows.append({"raw_scene_num": int(source.raw_scene_num), "scene_index": paper_scene_index(source.raw_scene_num),
                         "bin_low_um": low, "bin_high_um": high, "diameter_center_um": (low + high) / 2,
                         "first_10904_frame_count": int(early[i]), "last_13_frame_count": int(tail[i]),
                         "recorded_particle_count": int(early[i] + tail[i]), "recorded_frames": RECORDED_FRAMES})
        audits.append({"raw_scene_num": int(source.raw_scene_num), "source_name": path.name,
                       "source_bytes": path.stat().st_size, "source_sha256": sha256(path), **audit})
        print(f"Scene {source.raw_scene_num}: {int((early + tail).sum())} in-range detections, including {int(tail.sum())} in final 13 frames", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps({"schema_version": 1, "counts_sha256": sha256(args.output),
        "histogram_rule": "25 to 300 um in 5 um bins; final right edge included, as in original NumPy histogram",
        "sources": audits}, indent=2) + "\n")


if __name__ == "__main__":
    main()
