#!/usr/bin/env python3
"""Replay frozen tables or rerun collision inference without image downloads."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ra3d import minip
from ra3d.cascade import predict_full
from ra3d.evaluation import event_curve
from ra3d.io import read_table, write_json, write_table
from ra3d.models import load_model

EXPECTED_AP = {"3d_terminal": 0.9171254959438848,
               "2d_terminal": 0.9011106311866217, "minip2d_terminal": 0.858187450125321}


def compare(actual, reference, score_column):
    """Compare the same source candidates and corrected event estimates."""
    key = "_source_index"
    if not actual[key].is_unique or not reference[key].is_unique:
        raise AssertionError("Source candidate indices must be unique")
    left = actual.sort_values(key).reset_index(drop=True)
    right = reference.sort_values(key).reset_index(drop=True)
    same_keys = np.array_equal(left[key].to_numpy(), right[key].to_numpy())
    result = {"actual_events": len(actual), "reference_events": len(reference), "same_source_candidates": same_keys}
    if not same_keys:
        result["status"] = "MISMATCH"
        return result
    differences = {}
    for name in ["vanished_track_id", "survivor_track_id", "candidate_frame", "x_um", "y_um", "z_um"]:
        if name in left and name in right:
            differences[name] = float(np.max(np.abs(left[name].to_numpy(float) - right[name].to_numpy(float))))
    differences["score"] = float(np.max(np.abs(left.score.to_numpy(float) - right[score_column].to_numpy(float))))
    result["max_absolute_differences"] = differences
    # The released 3D model is a verified refit; single percentile-rank steps can
    # differ between the historical GPU reduction and CPU replay. Event identity,
    # integer time and corrected coordinates are checked separately.
    result["score_atol"] = 1e-5
    result["coordinate_atol_um"] = 1e-7
    result["status"] = "PASS" if all(differences[k] <= (1e-5 if k == "score" else 1e-7) for k in differences) else "MISMATCH"
    return result


def terminal(output, device):
    result = {}
    truth = read_table(ROOT / "data/labels/validation/seed260745/collision_events.parquet")
    for method in ["3d", "2d", "minip2d"]:
        if method == "3d":
            model = load_model(ROOT / "data/models/paper", device=device)
            full = read_table(ROOT / "data/features/validation/seed260745/paper92.parquet")
            pred, audit = predict_full(full, model, scene_id=260745)
        else:
            directory = "baseline2d" if method == "2d" else "minip2d"
            model = minip.load(ROOT / f"data/models/{directory}", device=device)
            full = read_table(ROOT / f"data/{directory}/validation/seed260745/full.parquet")
            pred = minip.predict(full, model, scene_id=260745)
            audit = {"trajectory_features": 49, "image_features": 15, "full_features": 64}
        reference = read_table(ROOT / f"data/predictions/{method}/validation/seed260745.parquet")
        curve, metric = event_curve({260745: pred}, {260745: truth})
        np.testing.assert_allclose(metric["event_ap"], EXPECTED_AP[f"{method}_terminal"], rtol=0, atol=1e-12)
        comparison = compare(pred, reference, "relation_attention_3d" if method == "3d" else "relation_attention_2d")
        write_table(pred, output / f"terminal_{method}.parquet")
        write_table(curve, output / f"terminal_{method}_pr.parquet")
        result[method] = {"comparison": comparison, "metrics": metric, "audit": audit}
        print(method, comparison["status"], metric, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["terminal"], default="terminal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=4)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Choose a new output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.cpu_threads)
    result = {}
    with threadpool_limits(limits=a.cpu_threads):
        result["terminal"] = terminal(a.output / "terminal", a.device)
    write_json(result, a.output / "verification.json")
    if '"MISMATCH"' in json.dumps(result):
        raise SystemExit("Prediction differences exceed the documented tolerance; see verification.json")
    print("PASS:", a.output / "verification.json", flush=True)


if __name__ == "__main__":
    main()
