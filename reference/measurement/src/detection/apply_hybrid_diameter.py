#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add final diameter columns using the selected SliceDiamModel fallback rule.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics-output", type=Path, default=None)
    p.add_argument("--ratio-threshold", type=float, default=1.05)
    p.add_argument("--slice-cap-um", type=float, default=100.0)
    p.add_argument("--pixel-pitch-um", type=float, default=10.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    df = pd.read_csv(args.input)
    required = {"w", "h", "diameter_um", "slice_diam_pred_um"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{args.input} is missing required columns: {missing}")

    if "diameter_px" in df.columns:
        pitch = np.where(df["diameter_px"].to_numpy(float) > 0, df["diameter_um"].to_numpy(float) / df["diameter_px"].to_numpy(float), args.pixel_pitch_um)
    else:
        pitch = np.full(len(df), args.pixel_pitch_um, dtype=np.float64)
    bbox_avg_um = 0.5 * (df["w"].to_numpy(float) + df["h"].to_numpy(float)) * pitch
    slice_pred = df["slice_diam_pred_um"].to_numpy(float)
    current = df["diameter_um"].to_numpy(float)
    fallback = slice_pred > args.ratio_threshold * np.maximum(bbox_avg_um, 1e-9)
    if math.isfinite(args.slice_cap_um):
        fallback &= slice_pred < args.slice_cap_um
    final = np.where(fallback, current, slice_pred)

    out = df.copy()
    out["bbox_avg_side_um"] = bbox_avg_um
    out["diameter_fallback_to_contrast_area"] = fallback.astype(np.int8)
    out["final_diameter_um"] = final
    out["final_diameter_source"] = np.where(fallback, "contrast_area_fallback", "slice_diammodel")
    out["final_diameter_rule"] = f"slice_pred > {args.ratio_threshold:g} * bbox_avg_side_um and slice_pred < {args.slice_cap_um:g}"
    out["x_um"] = out["seg_xc"].astype(float) * args.pixel_pitch_um
    out["y_um"] = out["seg_yc"].astype(float) * args.pixel_pitch_um
    if "slice" in out.columns:
        out["z_um"] = out["slice"].astype(float) * 100.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    metrics = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(out)),
        "ratio_threshold": float(args.ratio_threshold),
        "slice_cap_um": float(args.slice_cap_um),
        "fallback_count": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()) if len(fallback) else 0.0,
    }
    metrics_path = args.metrics_output or args.output.with_suffix(".hybrid_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
