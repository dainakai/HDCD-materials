"""Reproducible analytic trajectories with missing observations and axial outliers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Trajectory


def make_synthetic(
    *, seed: int = 260712, tracks: int = 8, points: int = 80, outlier_fraction: float = 0.08
) -> tuple[list[Trajectory], pd.DataFrame]:
    if tracks < 1 or points < 8 or not 0 <= outlier_fraction < 1:
        raise ValueError("tracks >= 1, points >= 8 and 0 <= outlier_fraction < 1 are required")
    rng = np.random.default_rng(seed)
    trajectories, truth = [], []
    for track_id in range(tracks):
        frames = np.arange(points)
        if track_id % 2:
            frames = frames[frames % 17 != 9]
        t = frames.astype(float)
        vx, vy = 0.15 + 0.02 * track_id, -0.08 - 0.01 * track_id
        curvature = 0.0015 * (-1) ** track_id
        x, y = 80 + 60 * track_id + vx * t, 140 + 45 * track_id + vy * t
        z = 150 + 80 * track_id + 0.35 * t + curvature * t**2
        noise = rng.normal(0, 0.65, len(t))
        outliers = rng.random(len(t)) < outlier_fraction
        noise[outliers] += rng.choice([-1, 1], outliers.sum()) * rng.uniform(15, 45, outliers.sum())
        diameter = np.full(len(t), 5 + track_id)
        trajectories.append(
            Trajectory(
                track_id,
                frames,
                x + rng.normal(0, 0.12, len(t)),
                y + rng.normal(0, 0.12, len(t)),
                z + noise,
                diameter,
            )
        )
        truth.append(
            pd.DataFrame(
                {
                    "track_id": track_id,
                    "frame": frames,
                    "x": x,
                    "y": y,
                    "z": z,
                    "vx": vx,
                    "vy": vy,
                    "vz": 0.35 + 2 * curvature * t,
                    "injected_outlier": outliers,
                }
            )
        )
    return trajectories, pd.concat(truth, ignore_index=True)


def write_synthetic(destination: str | Path, **kwargs) -> Path:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Synthetic data destination already contains files: {destination}")
    trajectories, truth = make_synthetic(**kwargs)
    path = destination / "trajectory3d"
    path.mkdir(parents=True, exist_ok=True)
    tables = []
    for t in trajectories:
        raw = pd.DataFrame(
            {"frame": t.frames, "xc": t.x, "yc": t.y, "slice": t.z, "diameter_px": t.diameter_px}
        )
        raw.to_csv(path / t.filename, index=False)
        raw["track_id"] = t.track_id
        tables.append(raw)
    pd.concat(tables, ignore_index=True).to_csv(destination / "tracks_with_diam_depth.csv", index=False)
    truth.to_csv(destination / "truth.csv", index=False)
    (destination / "synthetic.json").write_text(
        json.dumps({"generator": "analytic-trajectory-v1", **kwargs}, indent=2) + "\n"
    )
    return destination


def evaluate_tracks(prediction: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Evaluate aligned temporal IDs on exactly matching (track_id, frame) keys.

    Truth association must be established by the caller for real linked data.
    Rows are never dropped silently and truth is never used to fit the model.
    """
    keys = ["track_id", "frame"]
    for name, frame in (("prediction", prediction), ("truth", truth)):
        if not set(keys).issubset(frame) or frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError(f"{name} requires unique, non-missing (track_id, frame) keys")
    if prediction.empty or set(map(tuple, prediction[keys].to_numpy())) != set(
        map(tuple, truth[keys].to_numpy())
    ):
        raise ValueError("Prediction and truth must have exactly matching (track_id, frame) keys")
    pred = prediction.set_index(keys).sort_index()
    target = truth.set_index(keys).sort_index()
    metrics = {"points": len(pred), "tracks": int(pred.index.get_level_values(0).nunique())}
    for axis in ("x", "y", "z", "vx", "vy", "vz"):
        pcol = f"{axis}_smooth" if f"{axis}_smooth" in pred else axis
        if pcol not in pred or axis not in target:
            if axis in ("x", "y", "z"):
                raise ValueError(f"Missing {axis} coordinate in prediction or truth")
            continue
        delta = pred[pcol].to_numpy(float) - target[axis].to_numpy(float)
        if not np.isfinite(delta).all():
            raise ValueError(f"Non-finite {axis} values in prediction or truth")
        metrics[f"{axis}_rmse"] = float(np.sqrt(np.mean(delta**2)))
        if f"{axis}_raw" in pred:
            residual = pred[f"{axis}_raw"].to_numpy(float) - target[axis].to_numpy(float)
            if not np.isfinite(residual).all():
                raise ValueError(f"Non-finite raw {axis} coordinates")
            metrics[f"{axis}_raw_rmse"] = float(np.sqrt(np.mean(residual**2)))
    return metrics
