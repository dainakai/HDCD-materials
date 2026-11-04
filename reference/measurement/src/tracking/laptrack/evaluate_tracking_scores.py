#!/usr/bin/env python3
"""Evaluate synthetic droplet trajectory estimates against truth labels.

This script focuses on the first two task metrics requested for the
260617_synth benchmark:

1. velocity error
2. diameter error

    It first matches truth and estimated particles frame-by-frame using position,
then computes residuals on the matched pairs. Both raw reconstructed
``trajectory3d`` files and ``smoothedtrajs`` files are supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_CLEAN_DIR = REPO_ROOT / "clean"
TRACK_RE = re.compile(r"^path_([0-9]+)\.csv$")


@dataclass
class Point:
    frame: int
    track_id: int
    x: float
    y: float
    z: float
    diameter_px: float
    vx: float = math.nan
    vy: float = math.nan
    vz: float = math.nan
    match_x: float = math.nan
    match_y: float = math.nan
    match_z: float = math.nan
    center_in_volume: int = 1
    source: str = ""


@dataclass
class Dataset:
    name: str
    points: list[Point]
    frames: dict[int, list[int]]
    track_lengths: Counter
    source: str
    velocity_source: str


@dataclass
class Match:
    frame: int
    truth_index: int
    pred_index: int
    cost: float
    dx_px: float
    dy_px: float
    dz_slice: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR))
    parser.add_argument("--truth-csv", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--trajectory-dir", default="")
    parser.add_argument("--smoothed-dir", default="")
    parser.add_argument(
        "--smoothed-match-trajectory-dir",
        default="",
        help=(
            "Optional trajectory3d directory whose contrast-area centers are used only for matching "
            "smoothedtrajs rows. Metrics still use smoothed positions."
        ),
    )
    parser.add_argument(
        "--which",
        choices=("reconstructed", "smoothed", "both"),
        default="both",
        help="Which estimate set to evaluate.",
    )
    parser.add_argument(
        "--truth-scope",
        choices=("center", "all"),
        default="center",
        help="center uses only truth rows whose particle center is inside the observation volume.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dx-um", type=float, default=10.0)
    parser.add_argument("--dz-um", type=float, default=100.0)
    parser.add_argument("--fps", type=float, default=4000.0)
    parser.add_argument("--xy-gate-px", type=float, default=8.0)
    parser.add_argument("--z-gate-slice", type=float, default=80.0)
    parser.add_argument(
        "--match-space",
        choices=("xyz", "xy"),
        default="xyz",
        help="Coordinate space used for frame-wise truth/pred matching. xy ignores z completely.",
    )
    parser.add_argument(
        "--diameter-gate-px",
        type=float,
        default=0.0,
        help="Optional absolute diameter gate. 0 disables diameter gating.",
    )
    parser.add_argument(
        "--diameter-cost-weight",
        type=float,
        default=0.0,
        help="Optional diameter term weight for matching cost. 0 matches by position only.",
    )
    parser.add_argument(
        "--diameter-bins-um",
        default="30,50,75,100,125,150,200,250,300",
        help="Comma-separated truth-diameter bin edges for binned diameter-error summaries.",
    )
    parser.add_argument("--max-normalized-cost", type=float, default=1.0)
    parser.add_argument(
        "--pred-track-best-truth-metric",
        choices=("xy", "xyz"),
        default="xy",
        help=(
            "Metric used to choose the nearest truth trajectory for each predicted trajectory. "
            "Residual summaries are still reported for xy, z, and xyz."
        ),
    )
    parser.add_argument(
        "--pred-track-best-min-overlap-points",
        type=int,
        default=1,
        help="Minimum overlapping frames required when comparing one predicted trajectory to one truth trajectory.",
    )
    parser.add_argument(
        "--write-residuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-match residual CSV files.",
    )
    return parser.parse_args()


def parse_bin_edges(value: str) -> list[float]:
    edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(edges) < 2:
        raise ValueError("--diameter-bins-um must contain at least two comma-separated values")
    if any(not math.isfinite(edge) for edge in edges):
        raise ValueError("--diameter-bins-um contains a non-finite value")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--diameter-bins-um must be strictly increasing")
    return edges


def require_columns(fieldnames: list[str] | None, required: Iterable[str], path: Path) -> None:
    names = set(fieldnames or [])
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def parse_int(value: str) -> int:
    return int(float(value))


def parse_float(value: str) -> float:
    return float(value)


def match_x(point: Point) -> float:
    return point.match_x if math.isfinite(point.match_x) else point.x


def match_y(point: Point) -> float:
    return point.match_y if math.isfinite(point.match_y) else point.y


def match_z(point: Point) -> float:
    return point.match_z if math.isfinite(point.match_z) else point.z


def build_index(points: list[Point], name: str, source: str, velocity_source: str) -> Dataset:
    frames: dict[int, list[int]] = defaultdict(list)
    track_lengths: Counter = Counter()
    for i, point in enumerate(points):
        frames[point.frame].append(i)
        track_lengths[point.track_id] += 1
    return Dataset(name, points, dict(frames), track_lengths, source, velocity_source)


def load_truth(path: Path, *, truth_scope: str, dx_um: float, dz_um: float, fps: float) -> Dataset:
    points: list[Point] = []
    required = (
        "frame",
        "particle_id",
        "x_px",
        "y_px",
        "z_slice",
        "diameter_px",
        "vx_um_s",
        "vy_um_s",
        "vz_um_s",
    )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, required, path)
        has_center = "center_in_volume" in (reader.fieldnames or [])
        for row in reader:
            center = parse_int(row["center_in_volume"]) if has_center else 1
            if truth_scope == "center" and center != 1:
                continue
            points.append(
                Point(
                    frame=parse_int(row["frame"]),
                    track_id=parse_int(row["particle_id"]),
                    x=parse_float(row["x_px"]),
                    y=parse_float(row["y_px"]),
                    z=parse_float(row["z_slice"]),
                    diameter_px=parse_float(row["diameter_px"]),
                    vx=parse_float(row["vx_um_s"]) / (dx_um * fps),
                    vy=parse_float(row["vy_um_s"]) / (dx_um * fps),
                    vz=parse_float(row["vz_um_s"]) / (dz_um * fps),
                    center_in_volume=center,
                    source=row.get("file", ""),
                )
            )
    return build_index(points, "truth", str(path), "truth_labels")


def track_id_from_path(path: Path) -> int:
    match = TRACK_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a path_*.csv file: {path}")
    return int(match.group(1))


def iter_track_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"track directory not found: {directory}")
    files = sorted((p for p in directory.iterdir() if TRACK_RE.match(p.name)), key=lambda p: track_id_from_path(p))
    if not files:
        raise FileNotFoundError(f"no path_*.csv files found in {directory}")
    return files


def finite_difference_velocities(points: list[Point]) -> None:
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, point in enumerate(points):
        by_track[point.track_id].append(i)

    for indices in by_track.values():
        indices.sort(key=lambda i: points[i].frame)
        if len(indices) < 2:
            continue
        for pos, idx in enumerate(indices):
            if pos == 0:
                left = indices[pos]
                right = indices[pos + 1]
            elif pos == len(indices) - 1:
                left = indices[pos - 1]
                right = indices[pos]
            else:
                left = indices[pos - 1]
                right = indices[pos + 1]
            dt = points[right].frame - points[left].frame
            if dt <= 0:
                continue
            point = points[idx]
            point.vx = (points[right].x - points[left].x) / dt
            point.vy = (points[right].y - points[left].y) / dt
            point.vz = (points[right].z - points[left].z) / dt


def load_reconstructed(directory: Path) -> Dataset:
    points: list[Point] = []
    for path in iter_track_files(directory):
        track_id = track_id_from_path(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            require_columns(reader.fieldnames, ("frame", "xc", "yc", "slice", "diameter_px"), path)
            for row in reader:
                points.append(
                    Point(
                        frame=parse_int(row["frame"]),
                        track_id=track_id,
                        x=parse_float(row["xc"]),
                        y=parse_float(row["yc"]),
                        z=parse_float(row["slice"]),
                        diameter_px=parse_float(row["diameter_px"]),
                        source=path.name,
                    )
                )
    finite_difference_velocities(points)
    return build_index(points, "reconstructed", str(directory), "finite_difference_from_trajectory3d")


def load_match_trajectory_centers(directory: Path) -> dict[tuple[int, int], tuple[float, float, float]]:
    centers: dict[tuple[int, int], tuple[float, float, float]] = {}
    for path in iter_track_files(directory):
        track_id = track_id_from_path(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            require_columns(reader.fieldnames, ("frame", "xc", "yc", "slice"), path)
            for row in reader:
                key = (track_id, parse_int(row["frame"]))
                centers[key] = (
                    parse_float(row["xc"]),
                    parse_float(row["yc"]),
                    parse_float(row["slice"]),
                )
    return centers


def load_smoothed(directory: Path, *, match_trajectory_dir: Path | None = None) -> Dataset:
    points: list[Point] = []
    velocity_columns_present = True
    match_centers = load_match_trajectory_centers(match_trajectory_dir) if match_trajectory_dir is not None else {}
    for path in iter_track_files(directory):
        track_id = track_id_from_path(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            require_columns(reader.fieldnames, ("frame", "x", "y", "z", "diameter_px"), path)
            has_velocity = {"vx", "vy", "vz"}.issubset(set(reader.fieldnames or []))
            velocity_columns_present = velocity_columns_present and has_velocity
            for row in reader:
                frame = parse_int(row["frame"])
                mx, my, mz = match_centers.get((track_id, frame), (math.nan, math.nan, math.nan))
                points.append(
                    Point(
                        frame=frame,
                        track_id=track_id,
                        x=parse_float(row["x"]),
                        y=parse_float(row["y"]),
                        z=parse_float(row["z"]),
                        diameter_px=parse_float(row["diameter_px"]),
                        vx=parse_float(row["vx"]) if has_velocity else math.nan,
                        vy=parse_float(row["vy"]) if has_velocity else math.nan,
                        vz=parse_float(row["vz"]) if has_velocity else math.nan,
                        match_x=mx,
                        match_y=my,
                        match_z=mz,
                        source=path.name,
                    )
                )
    velocity_source = "smoothed_velocity_columns"
    if not velocity_columns_present:
        finite_difference_velocities(points)
        velocity_source = "finite_difference_from_smoothed_positions"
    source = str(directory)
    if match_trajectory_dir is not None:
        source += f" (match_centers={match_trajectory_dir})"
    return build_index(points, "smoothed", source, velocity_source)


def normalized_match_cost(
    truth_points: list[Point],
    pred_points: list[Point],
    *,
    xy_gate_px: float,
    z_gate_slice: float,
    match_space: str,
    diameter_gate_px: float,
    diameter_cost_weight: float,
) -> np.ndarray:
    tx = np.asarray([match_x(p) for p in truth_points], dtype=np.float64)[:, None]
    ty = np.asarray([match_y(p) for p in truth_points], dtype=np.float64)[:, None]
    tz = np.asarray([match_z(p) for p in truth_points], dtype=np.float64)[:, None]
    td = np.asarray([p.diameter_px for p in truth_points], dtype=np.float64)[:, None]
    px = np.asarray([match_x(p) for p in pred_points], dtype=np.float64)[None, :]
    py = np.asarray([match_y(p) for p in pred_points], dtype=np.float64)[None, :]
    pz = np.asarray([match_z(p) for p in pred_points], dtype=np.float64)[None, :]
    pd = np.asarray([p.diameter_px for p in pred_points], dtype=np.float64)[None, :]

    dx = px - tx
    dy = py - ty
    dz = pz - tz
    dd = pd - td

    cost_sq = (dx / xy_gate_px) ** 2 + (dy / xy_gate_px) ** 2
    valid = (np.abs(dx) <= xy_gate_px) & (np.abs(dy) <= xy_gate_px)

    if match_space == "xyz" and z_gate_slice > 0:
        cost_sq += (dz / z_gate_slice) ** 2
        valid &= np.abs(dz) <= z_gate_slice
    if diameter_gate_px > 0:
        valid &= np.abs(dd) <= diameter_gate_px
    if diameter_cost_weight > 0 and diameter_gate_px > 0:
        cost_sq += diameter_cost_weight * (dd / diameter_gate_px) ** 2

    cost = np.sqrt(cost_sq)
    cost[~valid] = 1e9
    return cost


def match_datasets(
    truth: Dataset,
    pred: Dataset,
    *,
    xy_gate_px: float,
    z_gate_slice: float,
    match_space: str,
    diameter_gate_px: float,
    diameter_cost_weight: float,
    max_normalized_cost: float,
) -> list[Match]:
    matches: list[Match] = []
    for frame in sorted(set(truth.frames) | set(pred.frames)):
        truth_indices = truth.frames.get(frame, [])
        pred_indices = pred.frames.get(frame, [])
        if not truth_indices or not pred_indices:
            continue
        truth_points = [truth.points[i] for i in truth_indices]
        pred_points = [pred.points[i] for i in pred_indices]
        cost = normalized_match_cost(
            truth_points,
            pred_points,
            xy_gate_px=xy_gate_px,
            z_gate_slice=z_gate_slice,
            match_space=match_space,
            diameter_gate_px=diameter_gate_px,
            diameter_cost_weight=diameter_cost_weight,
        )
        rows, cols = linear_sum_assignment(cost)
        for local_t, local_p in zip(rows, cols):
            c = float(cost[local_t, local_p])
            if c > max_normalized_cost:
                continue
            ti = truth_indices[int(local_t)]
            pi = pred_indices[int(local_p)]
            t = truth.points[ti]
            p = pred.points[pi]
            matches.append(
                Match(
                    frame=frame,
                    truth_index=ti,
                    pred_index=pi,
                    cost=c,
                    dx_px=match_x(p) - match_x(t),
                    dy_px=match_y(p) - match_y(t),
                    dz_slice=match_z(p) - match_z(t),
                )
            )
    return matches


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def error_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = finite(values)
    if arr.size == 0:
        return {
            "n": 0,
            "bias": None,
            "std": None,
            "mae": None,
            "rmse": None,
            "median": None,
            "median_abs": None,
            "p90_abs": None,
            "p95_abs": None,
            "p99_abs": None,
            "min": None,
            "max": None,
        }
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "bias": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mae": float(np.mean(abs_arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "median": float(np.median(arr)),
        "median_abs": float(np.median(abs_arr)),
        "p90_abs": float(np.percentile(abs_arr, 90)),
        "p95_abs": float(np.percentile(abs_arr, 95)),
        "p99_abs": float(np.percentile(abs_arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def binned_error_stats(
    truth_values: Iterable[float],
    errors: Iterable[float],
    bin_edges: list[float],
) -> list[dict[str, object]]:
    truth_arr = np.asarray(list(truth_values), dtype=np.float64)
    err_arr = np.asarray(list(errors), dtype=np.float64)
    valid = np.isfinite(truth_arr) & np.isfinite(err_arr)
    truth_arr = truth_arr[valid]
    err_arr = err_arr[valid]

    rows: list[dict[str, object]] = []
    for lo, hi in zip(bin_edges, bin_edges[1:]):
        in_bin = (lo <= truth_arr) & (truth_arr < hi)
        if hi == bin_edges[-1]:
            in_bin = (lo <= truth_arr) & (truth_arr <= hi)
        stats = error_stats(err_arr[in_bin])
        stats.update({
            "bin_min_um": float(lo),
            "bin_max_um": float(hi),
            "bin_label": f"[{lo:g}, {hi:g}{']' if hi == bin_edges[-1] else ')'}",
        })
        rows.append(stats)
    return rows


def scalar_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = finite(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def residual_percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = finite(values)
    if arr.size == 0:
        return {"n": 0, "mae": None, "p50": None, "p95": None}
    arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mae": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def track_frame_index(dataset: Dataset) -> dict[int, dict[int, int]]:
    by_track: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, point in enumerate(dataset.points):
        by_track[point.track_id][point.frame] = idx
    return by_track


def pred_track_best_truth_residuals(
    truth: Dataset,
    pred: Dataset,
    *,
    dx_um: float,
    dz_um: float,
    selection_metric: str,
    min_overlap_points: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    truth_by_track = track_frame_index(truth)
    pred_by_track = track_frame_index(pred)
    truth_ids_by_frame: dict[int, list[int]] = {}
    for frame, truth_indices in truth.frames.items():
        truth_ids_by_frame[frame] = [truth.points[i].track_id for i in truth_indices]

    rows: list[dict[str, object]] = []
    all_xy: list[float] = []
    all_abs_z: list[float] = []
    all_xyz: list[float] = []
    track_mean_xy: list[float] = []
    track_mean_abs_z: list[float] = []
    track_mean_xyz: list[float] = []
    overlap_points: list[int] = []
    pred_overlap_fractions: list[float] = []
    truth_overlap_fractions: list[float] = []
    tracks_without_candidate = 0

    for pred_id, pred_frames in pred_by_track.items():
        candidate_truth_ids: set[int] = set()
        for frame in pred_frames:
            candidate_truth_ids.update(truth_ids_by_frame.get(frame, ()))

        best: dict[str, object] | None = None
        best_key = math.inf
        for truth_id in candidate_truth_ids:
            truth_frames = truth_by_track.get(truth_id, {})
            common = sorted(set(pred_frames).intersection(truth_frames))
            if len(common) < min_overlap_points:
                continue

            select_xy_vals: list[float] = []
            select_xyz_vals: list[float] = []
            xy_vals: list[float] = []
            abs_z_vals: list[float] = []
            xyz_vals: list[float] = []
            for frame in common:
                p = pred.points[pred_frames[frame]]
                t = truth.points[truth_frames[frame]]
                select_dx_um = (match_x(p) - match_x(t)) * dx_um
                select_dy_um = (match_y(p) - match_y(t)) * dx_um
                select_dz_um = (match_z(p) - match_z(t)) * dz_um
                select_xy_vals.append(math.hypot(select_dx_um, select_dy_um))
                select_xyz_vals.append(
                    math.sqrt(select_dx_um * select_dx_um + select_dy_um * select_dy_um + select_dz_um * select_dz_um)
                )

                dx = (p.x - t.x) * dx_um
                dy = (p.y - t.y) * dx_um
                dz = (p.z - t.z) * dz_um
                xy_vals.append(math.hypot(dx, dy))
                abs_z_vals.append(abs(dz))
                xyz_vals.append(math.sqrt(dx * dx + dy * dy + dz * dz))

            # The best truth trajectory is selected by mean point residual over
            # the overlapping frames only. Coverage is reported separately.
            if selection_metric == "xy":
                key = float(np.mean(select_xy_vals))
            elif selection_metric == "xyz":
                key = float(np.mean(select_xyz_vals))
            else:
                raise ValueError(f"unknown selection_metric: {selection_metric}")
            if key < best_key:
                pred_len = pred.track_lengths[pred_id]
                truth_len = truth.track_lengths[truth_id]
                best_key = key
                best = {
                    "pred_id": int(pred_id),
                    "best_truth_id": int(truth_id),
                    "pred_len": int(pred_len),
                    "truth_len": int(truth_len),
                    "overlap_points": int(len(common)),
                    "pred_overlap_fraction": float(len(common) / pred_len) if pred_len else math.nan,
                    "truth_overlap_fraction": float(len(common) / truth_len) if truth_len else math.nan,
                    "mean_xy_norm_um": float(np.mean(xy_vals)),
                    "p50_xy_norm_um": float(np.percentile(xy_vals, 50)),
                    "p95_xy_norm_um": float(np.percentile(xy_vals, 95)),
                    "mean_abs_z_um": float(np.mean(abs_z_vals)),
                    "p50_abs_z_um": float(np.percentile(abs_z_vals, 50)),
                    "p95_abs_z_um": float(np.percentile(abs_z_vals, 95)),
                    "mean_xyz_norm_um": float(np.mean(xyz_vals)),
                    "p50_xyz_norm_um": float(np.percentile(xyz_vals, 50)),
                    "p95_xyz_norm_um": float(np.percentile(xyz_vals, 95)),
                    "_xy_vals": xy_vals,
                    "_abs_z_vals": abs_z_vals,
                    "_xyz_vals": xyz_vals,
                }

        if best is None:
            tracks_without_candidate += 1
            continue

        xy_vals = list(best.pop("_xy_vals"))  # type: ignore[arg-type]
        abs_z_vals = list(best.pop("_abs_z_vals"))  # type: ignore[arg-type]
        xyz_vals = list(best.pop("_xyz_vals"))  # type: ignore[arg-type]
        all_xy.extend(xy_vals)
        all_abs_z.extend(abs_z_vals)
        all_xyz.extend(xyz_vals)
        track_mean_xy.append(float(best["mean_xy_norm_um"]))
        track_mean_abs_z.append(float(best["mean_abs_z_um"]))
        track_mean_xyz.append(float(best["mean_xyz_norm_um"]))
        overlap_points.append(int(best["overlap_points"]))
        pred_overlap_fractions.append(float(best["pred_overlap_fraction"]))
        truth_overlap_fractions.append(float(best["truth_overlap_fraction"]))
        rows.append(best)

    summary = {
        "selection": {
            "metric": selection_metric,
            "min_overlap_points": int(min_overlap_points),
            "candidate_scope": "truth tracks with at least one point in the predicted trajectory frame range",
            "residual_scope": "overlapping frames between the predicted trajectory and the selected truth trajectory",
        },
        "counts": {
            "pred_tracks": int(len(pred_by_track)),
            "evaluated_pred_tracks": int(len(rows)),
            "tracks_without_candidate": int(tracks_without_candidate),
            "residual_points": int(len(all_xy)),
            "pred_points": int(len(pred.points)),
            "residual_point_fraction_of_pred_points": float(len(all_xy) / len(pred.points)) if pred.points else None,
        },
        "overlap": {
            "points_per_pred_track": scalar_stats(overlap_points),
            "pred_overlap_fraction": scalar_stats(pred_overlap_fractions),
            "truth_overlap_fraction": scalar_stats(truth_overlap_fractions),
        },
        "point_residual_um": {
            "xy_norm": residual_percentiles(all_xy),
            "abs_z": residual_percentiles(all_abs_z),
            "xyz_norm": residual_percentiles(all_xyz),
        },
        "per_pred_track_mean_residual_um": {
            "xy_norm": residual_percentiles(track_mean_xy),
            "abs_z": residual_percentiles(track_mean_abs_z),
            "xyz_norm": residual_percentiles(track_mean_xyz),
        },
    }
    rows.sort(key=lambda r: (int(r["pred_id"]), int(r["best_truth_id"])))
    return summary, rows


def px_velocity_to_um_s(vx: float, axis_scale_um: float, fps: float) -> float:
    return vx * axis_scale_um * fps


def pair_overlap_rows(truth: Dataset, pred: Dataset, matches: list[Match]) -> list[dict[str, object]]:
    counts: Counter = Counter()
    for match in matches:
        t = truth.points[match.truth_index]
        p = pred.points[match.pred_index]
        counts[(t.track_id, p.track_id)] += 1

    rows: list[dict[str, object]] = []
    for (truth_id, pred_id), n in counts.items():
        truth_len = truth.track_lengths[truth_id]
        pred_len = pred.track_lengths[pred_id]
        recall = n / truth_len if truth_len else 0.0
        purity = n / pred_len if pred_len else 0.0
        f1 = 2.0 * n / (truth_len + pred_len) if truth_len + pred_len else 0.0
        rows.append(
            {
                "truth_id": truth_id,
                "pred_id": pred_id,
                "matches": n,
                "truth_len": truth_len,
                "pred_len": pred_len,
                "recall": recall,
                "purity": purity,
                "f1": f1,
            }
        )
    rows.sort(key=lambda r: (-float(r["matches"]), int(r["truth_id"]), int(r["pred_id"])))
    return rows


def association_diagnostics(truth: Dataset, pred: Dataset, matches: list[Match]) -> dict[str, object]:
    matched_by_truth: dict[int, list[tuple[int, int]]] = defaultdict(list)
    matched_by_pred: dict[int, set[int]] = defaultdict(set)
    truth_to_pred_counts: dict[int, Counter] = defaultdict(Counter)
    pred_to_truth_counts: dict[int, Counter] = defaultdict(Counter)

    for match in matches:
        t = truth.points[match.truth_index]
        p = pred.points[match.pred_index]
        matched_by_truth[t.track_id].append((match.frame, p.track_id))
        matched_by_pred[p.track_id].add(t.track_id)
        truth_to_pred_counts[t.track_id][p.track_id] += 1
        pred_to_truth_counts[p.track_id][t.track_id] += 1

    id_switches = 0
    fragmented_truth_tracks = 0
    split_truth_tracks = 0
    merged_pred_tracks = 0
    for truth_id, seq in matched_by_truth.items():
        seq.sort()
        previous_pred: int | None = None
        switches_for_track = 0
        for _frame, pred_id in seq:
            if previous_pred is not None and pred_id != previous_pred:
                switches_for_track += 1
            previous_pred = pred_id
        id_switches += switches_for_track
        if switches_for_track > 0:
            fragmented_truth_tracks += 1
        if len(truth_to_pred_counts[truth_id]) > 1:
            split_truth_tracks += 1
    for pred_ids in pred_to_truth_counts.values():
        if len(pred_ids) > 1:
            merged_pred_tracks += 1

    truth_best_recall = []
    for truth_id, truth_len in truth.track_lengths.items():
        best = max(truth_to_pred_counts[truth_id].values(), default=0)
        truth_best_recall.append(best / truth_len if truth_len else math.nan)

    pred_best_purity = []
    for pred_id, pred_len in pred.track_lengths.items():
        best = max(pred_to_truth_counts[pred_id].values(), default=0)
        pred_best_purity.append(best / pred_len if pred_len else math.nan)

    return {
        "id_switches_from_frame_matches": int(id_switches),
        "truth_tracks_with_id_switch": int(fragmented_truth_tracks),
        "truth_tracks_split_across_multiple_pred_ids": int(split_truth_tracks),
        "pred_tracks_merging_multiple_truth_ids": int(merged_pred_tracks),
        "truth_best_single_pred_recall": scalar_stats(truth_best_recall),
        "pred_best_single_truth_purity": scalar_stats(pred_best_purity),
    }


def evaluate_matches(
    truth: Dataset,
    pred: Dataset,
    matches: list[Match],
    *,
    dx_um: float,
    dz_um: float,
    fps: float,
    diameter_bin_edges_um: list[float],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    residual_rows: list[dict[str, object]] = []
    truth_diameter_um_values: list[float] = []
    diameter_err_um: list[float] = []
    diameter_rel_err: list[float] = []
    dx_um_err: list[float] = []
    dy_um_err: list[float] = []
    dz_um_err: list[float] = []
    dvx_um_s: list[float] = []
    dvy_um_s: list[float] = []
    dvz_um_s: list[float] = []
    velocity_vector_error_um_s: list[float] = []
    speed_error_um_s: list[float] = []

    for match in matches:
        t = truth.points[match.truth_index]
        p = pred.points[match.pred_index]

        pred_diameter_um = p.diameter_px * dx_um
        truth_diameter_um = t.diameter_px * dx_um
        d_diameter_um = pred_diameter_um - truth_diameter_um
        truth_diameter_um_values.append(truth_diameter_um)
        diameter_err_um.append(d_diameter_um)
        diameter_rel_err.append(d_diameter_um / truth_diameter_um if truth_diameter_um > 0 else math.nan)

        pos_dx_um = (p.x - t.x) * dx_um
        pos_dy_um = (p.y - t.y) * dx_um
        pos_dz_um = (p.z - t.z) * dz_um
        dx_um_err.append(pos_dx_um)
        dy_um_err.append(pos_dy_um)
        dz_um_err.append(pos_dz_um)

        pred_vx_um_s = px_velocity_to_um_s(p.vx, dx_um, fps)
        pred_vy_um_s = px_velocity_to_um_s(p.vy, dx_um, fps)
        pred_vz_um_s = px_velocity_to_um_s(p.vz, dz_um, fps)
        truth_vx_um_s = px_velocity_to_um_s(t.vx, dx_um, fps)
        truth_vy_um_s = px_velocity_to_um_s(t.vy, dx_um, fps)
        truth_vz_um_s = px_velocity_to_um_s(t.vz, dz_um, fps)

        velocity_valid = all(
            math.isfinite(v)
            for v in (pred_vx_um_s, pred_vy_um_s, pred_vz_um_s, truth_vx_um_s, truth_vy_um_s, truth_vz_um_s)
        )
        dvx = pred_vx_um_s - truth_vx_um_s if velocity_valid else math.nan
        dvy = pred_vy_um_s - truth_vy_um_s if velocity_valid else math.nan
        dvz = pred_vz_um_s - truth_vz_um_s if velocity_valid else math.nan
        if velocity_valid:
            dvx_um_s.append(dvx)
            dvy_um_s.append(dvy)
            dvz_um_s.append(dvz)
            velocity_vector_error_um_s.append(math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz))
            pred_speed = math.sqrt(pred_vx_um_s**2 + pred_vy_um_s**2 + pred_vz_um_s**2)
            truth_speed = math.sqrt(truth_vx_um_s**2 + truth_vy_um_s**2 + truth_vz_um_s**2)
            speed_error_um_s.append(pred_speed - truth_speed)

        residual_rows.append(
            {
                "frame": match.frame,
                "truth_id": t.track_id,
                "pred_id": p.track_id,
                "match_cost": match.cost,
                "truth_x_px": t.x,
                "truth_y_px": t.y,
                "truth_z_slice": t.z,
                "pred_x_px": p.x,
                "pred_y_px": p.y,
                "pred_z_slice": p.z,
                "dx_um": pos_dx_um,
                "dy_um": pos_dy_um,
                "dz_um": pos_dz_um,
                "truth_diameter_um": truth_diameter_um,
                "pred_diameter_um": pred_diameter_um,
                "diameter_error_um": d_diameter_um,
                "diameter_relative_error": d_diameter_um / truth_diameter_um if truth_diameter_um > 0 else math.nan,
                "velocity_valid": int(velocity_valid),
                "truth_vx_um_s": truth_vx_um_s,
                "truth_vy_um_s": truth_vy_um_s,
                "truth_vz_um_s": truth_vz_um_s,
                "pred_vx_um_s": pred_vx_um_s,
                "pred_vy_um_s": pred_vy_um_s,
                "pred_vz_um_s": pred_vz_um_s,
                "dvx_um_s": dvx,
                "dvy_um_s": dvy,
                "dvz_um_s": dvz,
                "velocity_vector_error_um_s": math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) if velocity_valid else math.nan,
            }
        )

    pair_rows = pair_overlap_rows(truth, pred, matches)
    match_count = len(matches)
    truth_points = len(truth.points)
    pred_points = len(pred.points)
    summary = {
        "dataset": pred.name,
        "source": pred.source,
        "velocity_source": pred.velocity_source,
        "counts": {
            "truth_points": truth_points,
            "pred_points": pred_points,
            "matched_points": match_count,
            "unmatched_truth_points": truth_points - match_count,
            "unmatched_pred_points": pred_points - match_count,
            "match_recall": match_count / truth_points if truth_points else None,
            "match_precision": match_count / pred_points if pred_points else None,
        },
        "match_cost": scalar_stats(match.cost for match in matches),
        "position_error_um": {
            "x": error_stats(dx_um_err),
            "y": error_stats(dy_um_err),
            "z": error_stats(dz_um_err),
            "xy_norm": scalar_stats(math.hypot(x, y) for x, y in zip(dx_um_err, dy_um_err)),
            "xyz_norm": scalar_stats(
                math.sqrt(x * x + y * y + z * z) for x, y, z in zip(dx_um_err, dy_um_err, dz_um_err)
            ),
        },
        "diameter_error": {
            "um": error_stats(diameter_err_um),
            "relative": error_stats(diameter_rel_err),
            "by_truth_diameter_bin_um": binned_error_stats(
                truth_diameter_um_values,
                diameter_err_um,
                diameter_bin_edges_um,
            ),
            "relative_by_truth_diameter_bin_um": binned_error_stats(
                truth_diameter_um_values,
                diameter_rel_err,
                diameter_bin_edges_um,
            ),
        },
        "velocity_error_um_s": {
            "valid_matched_points": len(dvx_um_s),
            "x": error_stats(dvx_um_s),
            "y": error_stats(dvy_um_s),
            "z": error_stats(dvz_um_s),
            "vector_norm": scalar_stats(velocity_vector_error_um_s),
            "speed": error_stats(speed_error_um_s),
        },
        "association_diagnostics_from_matches": association_diagnostics(truth, pred, matches),
    }
    return summary, residual_rows, pair_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(
    truth: Dataset,
    pred: Dataset,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    matches = match_datasets(
        truth,
        pred,
        xy_gate_px=args.xy_gate_px,
        z_gate_slice=args.z_gate_slice,
        match_space=args.match_space,
        diameter_gate_px=args.diameter_gate_px,
        diameter_cost_weight=args.diameter_cost_weight,
        max_normalized_cost=args.max_normalized_cost,
    )
    summary, residual_rows, pair_rows = evaluate_matches(
        truth,
        pred,
        matches,
        dx_um=args.dx_um,
        dz_um=args.dz_um,
        fps=args.fps,
        diameter_bin_edges_um=parse_bin_edges(args.diameter_bins_um),
    )
    pred_track_summary, pred_track_rows = pred_track_best_truth_residuals(
        truth,
        pred,
        dx_um=args.dx_um,
        dz_um=args.dz_um,
        selection_metric=args.pred_track_best_truth_metric,
        min_overlap_points=max(1, int(args.pred_track_best_min_overlap_points)),
    )
    summary.update(
        {
            "truth_source": truth.source,
            "truth_scope": args.truth_scope,
            "matching": {
                "match_space": args.match_space,
                "xy_gate_px": args.xy_gate_px,
                "z_gate_slice": args.z_gate_slice,
                "diameter_gate_px": args.diameter_gate_px,
                "diameter_cost_weight": args.diameter_cost_weight,
                "diameter_bins_um": parse_bin_edges(args.diameter_bins_um),
                "max_normalized_cost": args.max_normalized_cost,
                "cost_definition": (
                    "sqrt((dx/xy_gate)^2 + (dy/xy_gate)^2"
                    + ("" if args.match_space == "xy" else " + (dz/z_gate)^2")
                    + (" + diameter_cost_weight*(dD/diameter_gate)^2" if args.diameter_cost_weight > 0 else "")
                    + ")"
                ),
            },
            "pred_track_best_truth_residuals": pred_track_summary,
            "units": {
                "positions": "x,y in px; z in reconstruction slices for matching; residuals in um",
                "diameter": "um",
                "velocity": "um/s",
                "dx_um": args.dx_um,
                "dz_um": args.dz_um,
                "fps": args.fps,
            },
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{pred.name}_scores.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / f"{pred.name}_pair_overlap.csv", pair_rows)
    write_csv(out_dir / f"{pred.name}_pred_track_best_truth_residuals.csv", pred_track_rows)
    if args.write_residuals:
        write_csv(out_dir / f"{pred.name}_matched_residuals.csv", residual_rows)
    return summary


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path | None]:
    clean_dir = Path(args.clean_dir).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else clean_dir / "runs" / "C000H001S0011"
    truth_csv = Path(args.truth_csv).resolve() if args.truth_csv else clean_dir / "labels" / "particles.csv"
    trajectory_dir = Path(args.trajectory_dir).resolve() if args.trajectory_dir else run_dir / "trajectory3d"
    smoothed_dir = Path(args.smoothed_dir).resolve() if args.smoothed_dir else run_dir / "smoothedtrajs"
    smoothed_match_trajectory_dir = (
        Path(args.smoothed_match_trajectory_dir).resolve() if args.smoothed_match_trajectory_dir else None
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else clean_dir / "evaluation"
    return truth_csv, trajectory_dir, smoothed_dir, out_dir, smoothed_match_trajectory_dir


def main() -> None:
    args = parse_args()
    truth_csv, trajectory_dir, smoothed_dir, out_dir, smoothed_match_trajectory_dir = resolve_paths(args)
    truth = load_truth(
        truth_csv,
        truth_scope=args.truth_scope,
        dx_um=args.dx_um,
        dz_um=args.dz_um,
        fps=args.fps,
    )

    summaries: dict[str, object] = {}
    if args.which in ("reconstructed", "both"):
        reconstructed = load_reconstructed(trajectory_dir)
        summaries["reconstructed"] = evaluate_one(truth, reconstructed, args, out_dir)
    if args.which in ("smoothed", "both"):
        smoothed = load_smoothed(smoothed_dir, match_trajectory_dir=smoothed_match_trajectory_dir)
        summaries["smoothed"] = evaluate_one(truth, smoothed, args, out_dir)

    combined = {
        "truth_source": truth.source,
        "truth_scope": args.truth_scope,
        "out_dir": str(out_dir),
        "results": summaries,
    }
    combined_path = out_dir / "tracking_scores_summary.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
