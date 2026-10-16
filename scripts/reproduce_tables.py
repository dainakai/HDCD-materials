#!/usr/bin/env python3
"""Recalculate paper Tables 3–4 from bundled candidates, predictions and trained weights."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from ra3d._core.context import context_frame
from ra3d._core.policy import temporal_mode_predictions
from ra3d._core.relation import relation_scores
from ra3d._core.residual import correct_predictions
from ra3d.cascade import ensemble_scores
from ra3d.evaluation import event_curve
from ra3d.io import read_table, write_json, write_table
from ra3d.models import load_model
from paper_table_layout import export_table

SEED = 260745
DATA = ROOT / "data"
REFERENCE = DATA / "paper_tables"


def summarize(predictions, truth, output, name, score_column="score"):
    curve, metrics = event_curve({SEED: predictions}, {SEED: truth}, score_column=score_column)
    curve["false_positive_events"] = curve.predicted_events - curve.matched_events
    write_table(curve, output / f"{name}_pr.parquet")
    result = {"method": name, "event_ap": metrics["event_ap"], "maximum_f1": metrics["maximum_f1"],
              "maximum_recall": metrics["maximum_recall"], "prediction_rows": len(predictions)}
    for target in (0.8, 0.9):
        point = metrics[f"operating_at_recall_{target:.2f}"]
        if point is None:
            raise AssertionError(f"{name}: target recall {target} is unreachable")
        suffix = str(target).replace(".", "p")
        result[f"precision_at_recall_{suffix}"] = point["precision"]
        result[f"false_positives_at_recall_{suffix}"] = point["false_positive_events"]
        result[f"threshold_at_recall_{suffix}"] = point["threshold"]
        result[f"achieved_recall_{suffix}"] = point["recall"]
    return result


def table3(truth, output):
    rows = []
    for method in ("3d", "2d", "minip2d"):
        pred = read_table(DATA / f"predictions/{method}/validation/seed{SEED}.parquet")
        row = summarize(pred, truth, output, method,
                        "relation_attention_3d" if method == "3d" else "relation_attention_2d")
        candidates = pd.read_parquet(DATA / f"evaluation/candidate_geometry_{method}.parquet")
        candidates["score"] = 1.0
        _, coverage = event_curve({SEED: candidates}, {SEED: truth})
        row.update(candidate_pool_rows=len(candidates), candidate_pool_recall=coverage["maximum_recall"])
        rows.append(row)
        print(f"Table 3, {method}: AP={row['event_ap']:.3f}, candidate recall={row['candidate_pool_recall']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    reference = pd.read_csv(REFERENCE / "table3_reference.csv", float_precision="round_trip")
    for column in reference.columns.drop("method"):
        np.testing.assert_allclose(frame[column], reference[column], rtol=0, atol=1e-12, err_msg=column)
    write_table(frame, output / "table3.csv")
    return {"status": "PASS", "rows": rows, "source": "frozen final predictions and uncorrected candidate pools",
            "paper_layout": export_table(frame, 3, output)}


def eventize(frame, scores, bundle, *, coordinate_suppression):
    candidates = frame.copy()
    candidates["stage_score"] = np.asarray(scores, dtype=float)
    policy = {**bundle["recipe"]["source_policy"], "score_column": "stage_score",
              "coordinate_suppression": coordinate_suppression}
    selected, audit = temporal_mode_predictions(candidates, seed=SEED, method="uncorrected", config=policy)
    pred = correct_predictions(selected, frame, bundle["residual_models"], source_method="uncorrected", method="score")
    return pred, audit


def table4(truth, output, device):
    bundle = load_model(DATA / "models/paper", device=device)
    rows, audits = [], {}
    static = read_table(DATA / f"features/validation/seed{SEED}/static_scored.parquet")
    # This is the historical screening/trajectory logit blend, before image
    # allocation. Keep all 566,261 candidates and their published row order.
    pred, audits["trajectory"] = eventize(static, static.logit_blend_static_75_top3, bundle,
                                         coordinate_suppression=False)
    rows.append(summarize(pred, truth, output, "trajectory"))
    write_table(pred, output / "trajectory_events.parquet")
    del static, pred
    gc.collect()
    full = read_table(DATA / f"features/validation/seed{SEED}/paper92.parquet")
    base = ensemble_scores(bundle["full_models"], full)
    np.testing.assert_allclose(base, np.load(DATA / f"scores/terminal_seed{SEED}.npy"), rtol=0, atol=1e-12)
    context = context_frame(full, base)
    ranks, relation_audit = relation_scores(context, bundle["relation_model"], bundle["relation_mean"],
                                             bundle["relation_std"], bundle["recipe"])
    for name, scores, suppression in [("image", base, False), ("relation", ranks, False),
                                      ("suppression", ranks, True)]:
        pred, audits[name] = eventize(full, scores, bundle, coordinate_suppression=suppression)
        rows.append(summarize(pred, truth, output, name))
        write_table(pred, output / f"{name}_events.parquet")
        print(f"Table 4, {name}: AP={rows[-1]['event_ap']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    reference = pd.read_csv(REFERENCE / "table4_reference.csv", float_precision="round_trip")
    for column in reference.columns.drop("method"):
        np.testing.assert_allclose(frame[column], reference[column], rtol=0, atol=1e-10, err_msg=column)
    write_table(frame, output / "table4.csv")
    return {"status": "PASS", "rows": rows, "selection": audits, "relation": relation_audit,
            "paper_layout": export_table(frame, 4, output),
            "source": "published trajectory scores, freshly inferred image/relation scores and residual corrections"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", choices=[3, 4], type=int, default=[3, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if output == ROOT or any(output.is_relative_to(ROOT / name) for name in ("data", "src", "reference", "vendor", "provenance")):
        parser.error("Choose an output under runs/ or outside the release inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    truth = read_table(DATA / f"labels/validation/seed{SEED}/collision_events.parquet")
    result = {}
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        if 3 in args.tables:
            result["table3"] = table3(truth, output)
        if 4 in args.tables:
            result["table4"] = table4(truth, output, args.device)
    write_json({"status": "PASS", **result}, output / "verification.json")
    print(f"PASS: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
