#!/usr/bin/env python3
"""Recompute and verify paper static features from bundled trajectories and candidate geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from ra3d.config import STATIC_FEATURES, STAGE1_FEATURES
from ra3d.features import static_features
from ra3d.io import read_table, write_table, write_json, legacy_candidate_csv
from ra3d.tracks import read_tracks, materialize_tracks

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=260745, choices=[260745])
    parser.add_argument("--max-candidates", type=int, default=500, help="0 computes the full shortlist")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_candidates < 0 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Use a nonnegative candidate limit and a new output directory")
    split = "validation" if args.seed == 260745 else "train"
    scene = f"{split}/seed{args.seed}"
    reference = read_table(ROOT / "data/features" / scene / "static77.parquet")
    if args.max_candidates:
        reference = reference.head(args.max_candidates)
    candidates = read_table(ROOT / "data/inputs/test_candidates.parquet").head(len(reference)).reset_index(drop=True)
    candidates = legacy_candidate_csv(candidates)
    tracks = read_tracks(ROOT / "data/tracking" / scene / "smoothedtrajs.parquet")
    needed = set(candidates.vanished_track_id) | set(candidates.survivor_track_id)
    tracks = tracks.loc[tracks.track_id.isin(needed)].reset_index(drop=True)
    directory = materialize_tracks(tracks, args.output / "tracks")
    computed = static_features(candidates, tracks, directory)
    deltas = {}
    for col in STATIC_FEATURES:
        actual, expected = computed[col].to_numpy(float), reference[col].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8, equal_nan=True, err_msg=col)
        finite = np.isfinite(actual) & np.isfinite(expected)
        deltas[col] = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
    write_table(computed, args.output / "static77.parquet")
    report = {"status": "PASS", "seed": args.seed, "candidate_rows": len(computed),
              "feature_columns_checked": len(STATIC_FEATURES), "uses_images": False,
              "candidate_feature_columns_retained": len(STAGE1_FEATURES),
              "additional_trajectory_columns_recomputed": len(STATIC_FEATURES) - len(STAGE1_FEATURES),
              "candidate_geometry": "published fixed shortlist", "trajectory_features": "recomputed",
              "max_absolute_difference": max(deltas.values()), "differences_by_feature": deltas}
    write_json(report, args.output / "verification.json")
    print(json.dumps({k: v for k, v in report.items() if k != "differences_by_feature"}, indent=2))


if __name__ == "__main__":
    main()
