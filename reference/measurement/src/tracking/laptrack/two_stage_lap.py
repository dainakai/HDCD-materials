#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


PROJECT_DIR = find_repo_root(Path(__file__).resolve())
INVALID = 1.0e6
EMBEDDINGS: np.ndarray | None = None


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def logit(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(np.asarray(x), 1e-6, 1.0 - 1e-6)
    return np.log(y / (1.0 - y))


def robust_scale(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return float(fallback)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad
    return float(scale if np.isfinite(scale) and scale > 1e-6 else fallback)


def robust_median(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.median(values))


def second_margin(cost: np.ndarray, axis: int) -> np.ndarray:
    finite = np.where(np.isfinite(cost) & (cost < INVALID * 0.5), cost, np.inf)
    if finite.shape[axis] < 2:
        shape = finite.shape[0] if axis == 1 else finite.shape[1]
        return np.full(shape, np.inf, dtype=np.float32)
    part = np.partition(finite, kth=1, axis=axis)
    best = np.take(part, 0, axis=axis)
    second = np.take(part, 1, axis=axis)
    with np.errstate(invalid="ignore"):
        margin = second - best
    margin[~np.isfinite(best)] = np.inf
    margin[~np.isfinite(second)] = np.inf
    return margin.astype(np.float32)


class UnionFind:
    def __init__(self, values: np.ndarray) -> None:
        self.parent = {int(v): int(v) for v in values}

    def find(self, x: int) -> int:
        p = self.parent[int(x)]
        if p != x:
            self.parent[int(x)] = self.find(p)
        return self.parent[int(x)]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return False
        root = min(ra, rb)
        other = max(ra, rb)
        self.parent[other] = root
        return True


@dataclass
class LinkParams:
    flow: np.ndarray
    motion_sigma: np.ndarray
    sigma_step: np.ndarray
    sigma_dv: np.ndarray


@dataclass
class Stage1Result:
    links: list[tuple[int, int, float]]
    tracklet_ids: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class TrackletState:
    root: int
    indices: np.ndarray
    start_frame: int
    end_frame: int
    length: int
    start_pos: np.ndarray
    end_pos: np.ndarray
    start_sigma: np.ndarray
    end_sigma: np.ndarray
    velocity: np.ndarray
    velocity_sigma: np.ndarray
    log_diam: float
    sigma_log_diam: float
    quality: float
    head_emb: np.ndarray | None = None
    tail_emb: np.ndarray | None = None


def boundary_probability(pos: np.ndarray, *, z_max_lap: float, radius: float, softness: float) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    margin = np.minimum.reduce([x, 1024.0 - x, y, 1024.0 - y, z, z_max_lap - z])
    return np.asarray(sigmoid((radius - margin) / max(softness, 1e-6)), dtype=np.float64)


def start_end_costs(df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    pos = df[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
    rho = np.clip(df["rho"].to_numpy(np.float64), 1e-5, 1.0 - 1e-5)
    p_bound = boundary_probability(
        pos,
        z_max_lap=float(args.z_slices) * 10.0,
        radius=float(args.boundary_radius),
        softness=float(args.boundary_softness),
    )
    p_miss = float(args.p_miss)
    p_not = 1.0 - (1.0 - p_miss) * (1.0 - p_bound)
    cost = -np.log(np.clip((1.0 - rho) + rho * p_not, 1e-9, 1.0))
    return cost.astype(np.float32), cost.astype(np.float32)


def estimate_global_flow(df: pd.DataFrame, sample_pairs: int = 80, min_rho: float = 0.20) -> tuple[np.ndarray, dict[str, Any]]:
    frames = sorted(df["frame"].unique())
    shifts: list[np.ndarray] = []
    for f0, f1 in zip(frames[:sample_pairs], frames[1 : sample_pairs + 1]):
        a_df = df[(df.frame == f0) & (df.rho >= min_rho)]
        b_df = df[(df.frame == f1) & (df.rho >= min_rho)]
        if len(a_df) < 30 or len(b_df) < 30:
            continue
        a_xy = a_df[["x_lap", "y_lap"]].to_numpy(np.float64)
        b_xy = b_df[["x_lap", "y_lap"]].to_numpy(np.float64)
        a_pos = a_df[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
        b_pos = b_df[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
        sample_step = max(1, len(a_xy) // 220)
        sample_xy = a_xy[::sample_step]
        sample_pos = a_pos[::sample_step]
        tree = cKDTree(b_xy)
        best: tuple[float, float, float] | None = None
        for dx in np.linspace(-3.5, 3.5, 29):
            for dy in np.linspace(0.0, 8.5, 35):
                d, idx = tree.query(sample_xy + np.array([dx, dy]), k=1, distance_upper_bound=10.0)
                finite = np.isfinite(d)
                if finite.sum() < max(12, len(sample_xy) * 0.55):
                    continue
                score = float(np.median(d[finite]) + 0.25 * np.percentile(d[finite], 90))
                if best is None or score < best[0]:
                    best = (score, float(dx), float(dy))
        if best is None:
            continue
        d, idx = tree.query(sample_xy + np.array([best[1], best[2]]), k=1, distance_upper_bound=10.0)
        finite = np.isfinite(d)
        if finite.sum() < 12:
            continue
        delta = b_pos[idx[finite]] - sample_pos[finite]
        shifts.append(np.median(delta, axis=0))
    if not shifts:
        flow = np.array([0.0, 4.0, 0.0], dtype=np.float64)
        return flow, {"estimated": False, "flow": flow.tolist()}
    arr = np.asarray(shifts, dtype=np.float64)
    flow = np.median(arr, axis=0)
    return flow, {
        "estimated": True,
        "sample_pairs": len(shifts),
        "flow": flow.tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
    }


def link_cost_matrix(
    a: pd.DataFrame,
    b: pd.DataFrame,
    params: LinkParams,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    apos = a[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
    bpos = b[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
    residual = bpos[None, :, :] - (apos[:, None, :] + params.flow[None, None, :])

    asig = a[["sigma_x", "sigma_y", "sigma_z"]].to_numpy(np.float64)
    bsig = b[["sigma_x", "sigma_y", "sigma_z"]].to_numpy(np.float64)
    svar = asig[:, None, :] ** 2 + bsig[None, :, :] ** 2 + params.motion_sigma[None, None, :] ** 2
    d2 = np.sum((residual * residual) / np.maximum(svar, 1e-6), axis=2)
    ref = np.maximum(np.median(svar.reshape(-1, 3), axis=0), 1e-6)
    logdet = np.sum(np.log(np.maximum(svar, 1e-6) / ref[None, None, :]), axis=2)
    pos_cost = 0.5 * (d2 + float(args.logdet_weight) * logdet)

    al = a["log_diameter_px"].to_numpy(np.float64)
    bl = b["log_diameter_px"].to_numpy(np.float64)
    sl = (
        a["sigma_log_diameter"].to_numpy(np.float64)[:, None] ** 2
        + b["sigma_log_diameter"].to_numpy(np.float64)[None, :] ** 2
        + float(args.sigma_log_diam_phys) ** 2
    )
    dl = bl[None, :] - al[:, None]
    dl2 = (dl * dl) / np.maximum(sl, 1e-8)
    diam_cost = 0.5 * (dl2 + float(args.logdet_weight) * np.log(np.maximum(sl, 1e-8) / np.median(sl)))

    rho_a = np.clip(a["rho"].to_numpy(np.float64), 1e-6, 1.0)
    rho_b = np.clip(b["rho"].to_numpy(np.float64), 1e-6, 1.0)
    rel_cost = -0.5 * float(args.lambda_rho_stage1) * (np.log(rho_a)[:, None] + np.log(rho_b)[None, :])

    total = pos_cost + float(args.diameter_weight_stage1) * diam_cost + rel_cost
    if EMBEDDINGS is not None and float(args.feature_weight_stage1) > 0:
        ai = a["embedding_index"].to_numpy(np.int64)
        bi = b["embedding_index"].to_numpy(np.int64)
        ea = EMBEDDINGS[ai]
        eb = EMBEDDINGS[bi]
        feat_dist = 1.0 - np.clip(ea @ eb.T, -1.0, 1.0)
        total += float(args.feature_weight_stage1) * np.clip(feat_dist / float(args.feature_scale), 0.0, 3.0)
    gate = (d2 <= float(args.stage1_pos_gate)) & (dl2 <= float(args.stage1_diam_gate))
    total = np.where(gate, total, np.inf).astype(np.float32)
    return total, d2.astype(np.float32)


def solve_adjacent_lap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    params: LinkParams,
    end_cost_all: np.ndarray,
    start_cost_all: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    cost, _d2 = link_cost_matrix(a, b, params, args)
    n, m = cost.shape
    diag = {"candidates": int(np.isfinite(cost).sum()), "accepted": 0, "ambiguous_rejected": 0}
    if n == 0 or m == 0:
        return [], diag
    row_obs = a["obs_id"].to_numpy(np.int64)
    col_obs = b["obs_id"].to_numpy(np.int64)
    end_cost = end_cost_all[row_obs]
    start_cost = start_cost_all[col_obs]
    delta = cost - end_cost[:, None] - start_cost[None, :]
    shifted_delta = delta + float(args.stage1_min_saving)
    margin_i = second_margin(shifted_delta, axis=1)
    margin_j = second_margin(shifted_delta, axis=0)
    assign = np.full((n, m + n), 0.0, dtype=np.float32)
    assign[:, :m] = np.where(np.isfinite(shifted_delta), shifted_delta, INVALID)
    rows, cols = linear_sum_assignment(assign)
    links: list[tuple[int, int, float]] = []
    for r, c in zip(rows, cols):
        if c >= m:
            continue
        value = float(assign[r, c])
        if value >= 0.0:
            continue
        if min(float(margin_i[r]), float(margin_j[c])) < float(args.stage1_margin):
            diag["ambiguous_rejected"] += 1
            continue
        links.append((int(row_obs[r]), int(col_obs[c]), float(cost[r, c])))
    diag["accepted"] = len(links)
    return links, diag


def tracklets_from_links(n: int, links: list[tuple[int, int, float]], df: pd.DataFrame) -> np.ndarray:
    uf = UnionFind(np.arange(n, dtype=np.int64))
    for i, j, _cost in links:
        uf.union(i, j)
    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)
    first_frame: dict[int, tuple[int, int]] = {}
    frames = df["frame"].to_numpy(np.int64)
    for idx, root in enumerate(roots):
        cur = first_frame.get(int(root))
        key = (int(frames[idx]), idx)
        if cur is None or key < cur:
            first_frame[int(root)] = key
    sorted_roots = sorted(first_frame, key=lambda r: first_frame[r])
    remap = {root: k + 1 for k, root in enumerate(sorted_roots)}
    return np.asarray([remap[int(root)] for root in roots], dtype=np.int32)


def normalize_seed_tracklets(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise RuntimeError(f"seed tracklet column not found: {column}")
    values = df[column].to_numpy(np.int64)
    first: dict[int, tuple[int, int]] = {}
    frames = df["frame"].to_numpy(np.int64)
    for idx, value in enumerate(values):
        key = (int(frames[idx]), idx)
        cur = first.get(int(value))
        if cur is None or key < cur:
            first[int(value)] = key
    order = sorted(first, key=lambda v: first[v])
    remap = {old: i + 1 for i, old in enumerate(order)}
    return np.asarray([remap[int(v)] for v in values], dtype=np.int32)


def run_stage1(df: pd.DataFrame, params: LinkParams, args: argparse.Namespace, label: str) -> Stage1Result:
    end_cost, start_cost = start_end_costs(df, args)
    frame_groups = {int(frame): g.copy() for frame, g in df.groupby("frame", sort=True)}
    frames = sorted(frame_groups)
    links: list[tuple[int, int, float]] = []
    counters: Counter = Counter()
    for f0, f1 in tqdm(list(zip(frames, frames[1:])), desc=f"Stage1 {label}", unit="pair"):
        pair_links, diag = solve_adjacent_lap(frame_groups[f0], frame_groups[f1], params, end_cost, start_cost, args)
        links.extend(pair_links)
        counters.update(diag)
    tracklet_ids = tracklets_from_links(len(df), links, df)
    lengths = pd.Series(tracklet_ids).value_counts().to_numpy()
    diagnostics = {
        "label": label,
        "links": len(links),
        "tracklets": int(len(lengths)),
        "length_quantiles": quantiles(lengths),
        "counters": dict(counters),
        "flow": params.flow.tolist(),
        "motion_sigma": params.motion_sigma.tolist(),
    }
    return Stage1Result(links=links, tracklet_ids=tracklet_ids, diagnostics=diagnostics)


def calibrate_from_links(df: pd.DataFrame, links: list[tuple[int, int, float]], params: LinkParams, args: argparse.Namespace) -> LinkParams:
    if len(links) < 20:
        return params
    a_idx = np.asarray([i for i, _j, _c in links], dtype=np.int64)
    b_idx = np.asarray([j for _i, j, _c in links], dtype=np.int64)
    apos = df.loc[a_idx, ["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
    bpos = df.loc[b_idx, ["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
    delta = bpos - apos
    rho = np.minimum(df.loc[a_idx, "rho"].to_numpy(float), df.loc[b_idx, "rho"].to_numpy(float))
    keep = rho >= np.quantile(rho, 0.35)
    if keep.sum() < 20:
        keep = np.ones(len(links), dtype=bool)
    flow = np.median(delta[keep], axis=0)
    resid = delta[keep] - flow[None, :]
    sig_a = df.loc[a_idx[keep], ["sigma_x", "sigma_y", "sigma_z"]].to_numpy(np.float64)
    sig_b = df.loc[b_idx[keep], ["sigma_x", "sigma_y", "sigma_z"]].to_numpy(np.float64)
    obs_var = np.median(sig_a * sig_a + sig_b * sig_b, axis=0)
    scale = np.asarray([robust_scale(resid[:, axis], params.motion_sigma[axis]) for axis in range(3)])
    motion_var = np.maximum(
        np.asarray(args.min_motion_sigma, dtype=np.float64) ** 2,
        scale * scale - obs_var,
    )
    motion_sigma = np.sqrt(motion_var)
    sigma_step = np.maximum(motion_sigma, np.asarray(args.min_motion_sigma, dtype=np.float64))
    sigma_dv = np.maximum(sigma_step * float(args.sigma_dv_scale), np.asarray(args.min_velocity_sigma, dtype=np.float64))
    return LinkParams(flow=flow, motion_sigma=motion_sigma, sigma_step=sigma_step, sigma_dv=sigma_dv)


def weighted_line(frames: np.ndarray, values: np.ndarray, sigma: np.ndarray, fallback_v: float) -> tuple[float, float, float, float, float]:
    if len(frames) <= 1 or frames[-1] == frames[0]:
        return float(values[0]), float(values[-1]), float(fallback_v), float(sigma[0] ** 2), 1.0e4
    weights = 1.0 / np.maximum(sigma * sigma, 1e-6)
    t = frames.astype(np.float64)
    y = values.astype(np.float64)
    tw = float(np.sum(weights * t) / np.sum(weights))
    yw = float(np.sum(weights * y) / np.sum(weights))
    dt = t - tw
    denom = float(np.sum(weights * dt * dt))
    if denom <= 1e-9:
        v = fallback_v
    else:
        v = float(np.sum(weights * dt * (y - yw)) / denom)
    b = yw - v * tw
    residual = y - (b + v * t)
    if len(frames) > 2:
        kappa = max(1.0, float(np.sum(weights * residual * residual) / max(1, len(frames) - 2)))
    else:
        kappa = 1.0
    start_var_fit = kappa * (1.0 / np.sum(weights) + (frames[0] - tw) ** 2 / max(denom, 1e-9))
    end_var_fit = kappa * (1.0 / np.sum(weights) + (frames[-1] - tw) ** 2 / max(denom, 1e-9))
    vel_var = kappa / max(denom, 1e-9)
    start_var = max(float(sigma[0] ** 2), float(start_var_fit))
    end_var = max(float(sigma[-1] ** 2), float(end_var_fit))
    return float(b + v * frames[0]), float(b + v * frames[-1]), float(v), start_var, float(max(vel_var, 1e-6)), float(end_var)


def estimate_tracklet_states(df: pd.DataFrame, tracklet_ids: np.ndarray, params: LinkParams) -> dict[int, TrackletState]:
    states: dict[int, TrackletState] = {}
    work = pd.DataFrame({"tracklet": tracklet_ids, "obs_index": np.arange(len(df), dtype=np.int64), "frame": df["frame"]})
    for root, rows in work.groupby("tracklet", sort=False):
        idx = rows.sort_values(["frame", "obs_index"])["obs_index"].to_numpy(np.int64)
        g = df.loc[idx]
        frames = g["frame"].to_numpy(np.float64)
        pos = g[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float64)
        sig = g[["sigma_x", "sigma_y", "sigma_z"]].to_numpy(np.float64)
        if len(idx) == 1:
            start_pos = end_pos = pos[0].copy()
            start_sigma = end_sigma = sig[0].copy()
            velocity = params.flow.copy()
            velocity_sigma = np.maximum(params.sigma_step * 8.0, np.array([12.0, 12.0, 80.0], dtype=np.float64))
        else:
            start_pos = np.zeros(3, dtype=np.float64)
            end_pos = np.zeros(3, dtype=np.float64)
            start_var = np.zeros(3, dtype=np.float64)
            end_var = np.zeros(3, dtype=np.float64)
            velocity = np.zeros(3, dtype=np.float64)
            vel_var = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                s0, s1, v, vv, ev, evar = weighted_line(frames, pos[:, axis], sig[:, axis], params.flow[axis])
                start_pos[axis] = s0
                end_pos[axis] = s1
                start_var[axis] = vv
                end_var[axis] = evar
                velocity[axis] = v
                vel_var[axis] = ev
            start_sigma = np.sqrt(np.maximum(start_var, sig[0] ** 2))
            end_sigma = np.sqrt(np.maximum(end_var, sig[-1] ** 2))
            velocity_sigma = np.sqrt(np.maximum(vel_var, np.array([0.05, 0.05, 0.10]) ** 2))
            if len(idx) == 2:
                velocity_sigma = np.maximum(
                    velocity_sigma,
                    np.sqrt((sig[0] ** 2 + sig[-1] ** 2 + params.motion_sigma**2) / max(1.0, (frames[-1] - frames[0]) ** 2)),
                )

        logd = g["log_diameter_px"].to_numpy(np.float64)
        slog = g["sigma_log_diameter"].to_numpy(np.float64)
        w = 1.0 / np.maximum(slog * slog, 1e-6)
        mean_logd = float(np.sum(w * logd) / np.sum(w))
        diam_resid = logd - mean_logd
        kappa = max(1.0, float(np.sum(w * diam_resid * diam_resid) / max(1, len(idx) - 1)))
        sigma_logd = float(math.sqrt(kappa / np.sum(w)))
        rho = np.clip(g["rho"].to_numpy(np.float64), 1e-6, 1.0 - 1e-6)
        q = float(sigmoid(np.mean(logit(rho)) + 0.65 * math.log1p(len(idx))))
        head_emb = tail_emb = None
        if EMBEDDINGS is not None:
            emb_idx = g["embedding_index"].to_numpy(np.int64)
            head_emb = normalize_vec(EMBEDDINGS[emb_idx[: min(5, len(emb_idx))]].mean(axis=0))
            tail_emb = normalize_vec(EMBEDDINGS[emb_idx[-min(5, len(emb_idx)) :]].mean(axis=0))
        states[int(root)] = TrackletState(
            root=int(root),
            indices=idx,
            start_frame=int(frames[0]),
            end_frame=int(frames[-1]),
            length=int(len(idx)),
            start_pos=start_pos,
            end_pos=end_pos,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            velocity=velocity,
            velocity_sigma=velocity_sigma,
            log_diam=mean_logd,
            sigma_log_diam=sigma_logd,
            quality=float(np.clip(q, 1e-5, 1.0 - 1e-5)),
            head_emb=head_emb,
            tail_emb=tail_emb,
        )
    return states


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-8)


def tracklet_boundary_costs(states: list[TrackletState], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    end_pos = np.stack([s.end_pos for s in states], axis=0)
    start_pos = np.stack([s.start_pos for s in states], axis=0)
    q = np.asarray([s.quality for s in states], dtype=np.float64)
    p_exit = boundary_probability(
        end_pos,
        z_max_lap=float(args.z_slices) * 10.0,
        radius=float(args.boundary_radius_stage2),
        softness=float(args.boundary_softness),
    )
    p_enter = boundary_probability(
        start_pos,
        z_max_lap=float(args.z_slices) * 10.0,
        radius=float(args.boundary_radius_stage2),
        softness=float(args.boundary_softness),
    )
    p_longmiss = float(args.p_longmiss)
    p_end = 1.0 - (1.0 - p_exit) * (1.0 - p_longmiss)
    p_start = 1.0 - (1.0 - p_enter) * (1.0 - p_longmiss)
    end_cost = -np.log(np.clip((1.0 - q) + q * p_end, 1e-9, 1.0))
    start_cost = -np.log(np.clip((1.0 - q) + q * p_start, 1e-9, 1.0))
    return end_cost.astype(np.float32), start_cost.astype(np.float32)


def stage2_pair_cost(a: TrackletState, b: TrackletState, params: LinkParams, args: argparse.Namespace) -> tuple[float, dict[str, float]] | None:
    if b.start_frame <= a.end_frame:
        return None
    n_steps = b.start_frame - a.end_frame
    missing = n_steps - 1
    if missing > int(args.stage2_max_gap):
        return None
    pred_ab = a.end_pos + a.velocity * n_steps
    pred_ba = b.start_pos - b.velocity * n_steps
    e_ab = b.start_pos - pred_ab
    e_ba = a.end_pos - pred_ba
    s_ab = a.end_sigma**2 + (n_steps * n_steps) * a.velocity_sigma**2 + b.start_sigma**2 + n_steps * params.sigma_step**2
    s_ba = b.start_sigma**2 + (n_steps * n_steps) * b.velocity_sigma**2 + a.end_sigma**2 + n_steps * params.sigma_step**2
    d2_ab = float(np.sum(e_ab * e_ab / np.maximum(s_ab, 1e-6)))
    d2_ba = float(np.sum(e_ba * e_ba / np.maximum(s_ba, 1e-6)))
    if min(d2_ab, d2_ba) > float(args.stage2_pos_gate_hard):
        return None
    if d2_ab > float(args.stage2_pos_gate) and d2_ba > float(args.stage2_pos_gate):
        return None
    wa = 1.0 / max(float(np.sum(a.velocity_sigma**2)), 1e-6)
    wb = 1.0 / max(float(np.sum(b.velocity_sigma**2)), 1e-6)
    wa = wa / max(wa + wb, 1e-6)
    wb = 1.0 - wa
    pos_cost_ab = 0.5 * (d2_ab + float(args.logdet_weight) * np.sum(np.log(np.maximum(s_ab, 1e-6))))
    pos_cost_ba = 0.5 * (d2_ba + float(args.logdet_weight) * np.sum(np.log(np.maximum(s_ba, 1e-6))))
    pos_cost = wa * pos_cost_ab + wb * pos_cost_ba

    ev = b.velocity - a.velocity
    sv = a.velocity_sigma**2 + b.velocity_sigma**2 + params.sigma_dv**2 * max(1, n_steps)
    vel_d2 = float(np.sum(ev * ev / np.maximum(sv, 1e-6)))
    vel_cost = 0.5 * (vel_d2 + float(args.logdet_weight) * np.sum(np.log(np.maximum(sv, 1e-6))))

    dl = b.log_diam - a.log_diam
    sl = a.sigma_log_diam**2 + b.sigma_log_diam**2 + float(args.sigma_log_diam_phys) ** 2
    diam_d2 = float(dl * dl / max(sl, 1e-8))
    if diam_d2 > float(args.stage2_diam_gate):
        return None
    diam_cost = 0.5 * (diam_d2 + float(args.logdet_weight) * math.log(max(sl, 1e-8)))
    gap_cost = float(args.stage2_gap_cost) * missing
    rel_cost = -0.5 * float(args.lambda_quality_stage2) * (math.log(a.quality) + math.log(b.quality))
    feat_cost = 0.0
    if (
        EMBEDDINGS is not None
        and float(args.feature_weight_stage2) > 0
        and a.tail_emb is not None
        and b.head_emb is not None
    ):
        feat_dist = 1.0 - float(np.clip(np.dot(a.tail_emb, b.head_emb), -1.0, 1.0))
        feat_cost = float(args.feature_weight_stage2) * min(3.0, feat_dist / float(args.feature_scale))
    total = (
        float(args.pred_weight_stage2) * pos_cost
        + float(args.velocity_weight_stage2) * vel_cost
        + float(args.diameter_weight_stage2) * diam_cost
        + gap_cost
        + rel_cost
        + feat_cost
    )
    if not np.isfinite(total):
        return None
    return float(total), {
        "n_steps": float(n_steps),
        "missing": float(missing),
        "d2_ab": d2_ab,
        "d2_ba": d2_ba,
        "vel_d2": vel_d2,
        "diam_d2": diam_d2,
        "pos_cost": float(pos_cost),
        "vel_cost": float(vel_cost),
        "diam_cost": float(diam_cost),
        "gap_cost": float(gap_cost),
        "rel_cost": float(rel_cost),
        "feat_cost": float(feat_cost),
    }


def run_stage2_once(
    states_map: dict[int, TrackletState],
    params: LinkParams,
    args: argparse.Namespace,
    pass_index: int,
) -> tuple[list[tuple[int, int, float, dict[str, float]]], dict[str, Any]]:
    states = sorted(states_map.values(), key=lambda s: (s.start_frame, s.end_frame, s.root))
    n = len(states)
    root_to_pos = {s.root: i for i, s in enumerate(states)}
    by_start: dict[int, list[TrackletState]] = defaultdict(list)
    for s in states:
        by_start[s.start_frame].append(s)
    end_cost, start_cost = tracklet_boundary_costs(states, args)
    assign = np.full((n, n + n), 0.0, dtype=np.float32)
    assign[:, :n] = INVALID
    info: dict[tuple[int, int], dict[str, float]] = {}
    candidate_count = 0
    for ai, a in enumerate(tqdm(states, desc=f"Stage2 candidates pass {pass_index}", unit="tracklet", leave=False)):
        for n_steps in range(1, int(args.stage2_max_gap) + 2):
            starts = by_start.get(a.end_frame + n_steps)
            if not starts:
                continue
            for b in starts:
                bi = root_to_pos[b.root]
                result = stage2_pair_cost(a, b, params, args)
                if result is None:
                    continue
                cost, detail = result
                delta = cost - float(end_cost[ai]) - float(start_cost[bi])
                shifted = delta + float(args.stage2_min_saving)
                if shifted < 0.0:
                    assign[ai, bi] = float(shifted)
                    info[(ai, bi)] = detail
                    candidate_count += 1
    rows, cols = linear_sum_assignment(assign)
    selected: list[tuple[int, int, float, dict[str, float]]] = []
    for r, c in zip(rows, cols):
        if c >= n:
            continue
        value = float(assign[r, c])
        if value >= 0.0:
            continue
        a = states[int(r)]
        b = states[int(c)]
        detail = dict(info[(int(r), int(c))])
        detail["delta"] = value
        selected.append((a.root, b.root, value, detail))
    return selected, {
        "pass": pass_index,
        "tracklets": n,
        "candidates": candidate_count,
        "selected": len(selected),
    }


def run_stage2_once_local(
    states_map: dict[int, TrackletState],
    params: LinkParams,
    args: argparse.Namespace,
    pass_index: int,
) -> tuple[list[tuple[int, int, float, dict[str, float]]], dict[str, Any]]:
    states = sorted(states_map.values(), key=lambda s: (s.start_frame, s.end_frame, s.root))
    by_start: dict[int, list[TrackletState]] = defaultdict(list)
    for s in states:
        by_start[s.start_frame].append(s)

    end_cost_map: dict[int, float] = {}
    start_cost_map: dict[int, float] = {}
    for frame_states in by_start.values():
        # Boundary costs only depend on the state, so computing them in chunks is fine.
        e, st = tracklet_boundary_costs(frame_states, args)
        for state, ec, sc in zip(frame_states, e, st):
            end_cost_map[state.root] = float(ec)
            start_cost_map[state.root] = float(sc)

    used_ends: set[int] = set()
    used_starts: set[int] = set()
    selected: list[tuple[int, int, float, dict[str, float]]] = []
    candidate_count = 0
    frames = sorted(by_start)
    for start_frame in tqdm(frames, desc=f"Stage2 local pass {pass_index}", unit="frame", leave=False):
        cols = [s for s in by_start[start_frame] if s.root not in used_starts]
        if not cols:
            continue
        rows = [
            s
            for s in states
            if s.root not in used_ends
            and s.end_frame < start_frame
            and start_frame - s.end_frame - 1 <= int(args.stage2_max_gap)
        ]
        if not rows:
            continue
        n = len(rows)
        m = len(cols)
        assign = np.full((n, m + n), 0.0, dtype=np.float32)
        assign[:, :m] = INVALID
        info: dict[tuple[int, int], dict[str, float]] = {}
        for ri, a in enumerate(rows):
            for ci, b in enumerate(cols):
                result = stage2_pair_cost(a, b, params, args)
                if result is None:
                    continue
                cost, detail = result
                delta = cost - end_cost_map[a.root] - start_cost_map[b.root]
                shifted = delta + float(args.stage2_min_saving)
                if shifted < 0.0:
                    assign[ri, ci] = float(shifted)
                    info[(ri, ci)] = detail
                    candidate_count += 1
        if not info:
            continue
        rids, cids = linear_sum_assignment(assign)
        for r, c in zip(rids, cids):
            if c >= m:
                continue
            value = float(assign[r, c])
            if value >= 0.0:
                continue
            a = rows[int(r)]
            b = cols[int(c)]
            if a.root in used_ends or b.root in used_starts:
                continue
            detail = dict(info[(int(r), int(c))])
            detail["delta"] = value
            selected.append((a.root, b.root, value, detail))
            used_ends.add(a.root)
            used_starts.add(b.root)
    return selected, {
        "pass": pass_index,
        "tracklets": len(states),
        "candidates": candidate_count,
        "selected": len(selected),
        "solver": "local_by_start_frame",
    }


def run_stage2(df: pd.DataFrame, tracklet_ids: np.ndarray, params: LinkParams, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    unique = np.unique(tracklet_ids)
    uf = UnionFind(unique)
    pass_diags: list[dict[str, Any]] = []
    all_merges: list[dict[str, Any]] = []
    for pass_index in range(1, int(args.stage2_passes) + 1):
        roots = np.asarray([uf.find(int(t)) for t in tracklet_ids], dtype=np.int32)
        states = estimate_tracklet_states(df, roots, params)
        if args.stage2_solver == "global":
            merges, diag = run_stage2_once(states, params, args, pass_index)
        else:
            merges, diag = run_stage2_once_local(states, params, args, pass_index)
        accepted = 0
        for a, b, delta, detail in merges:
            if uf.union(a, b):
                accepted += 1
                item = {"pass": pass_index, "from": int(a), "to": int(b), "delta": float(delta)}
                item.update({k: float(v) for k, v in detail.items()})
                all_merges.append(item)
        diag["accepted"] = accepted
        pass_diags.append(diag)
        if accepted == 0:
            break
    root_ids = np.asarray([uf.find(int(t)) for t in tracklet_ids], dtype=np.int32)
    order_df = pd.DataFrame({"root": root_ids, "frame": df["frame"].to_numpy(np.int64), "obs": np.arange(len(df))})
    order = order_df.groupby("root", sort=False).agg({"frame": "min", "obs": "min"}).sort_values(["frame", "obs"])
    remap = {int(root): i + 1 for i, root in enumerate(order.index.to_numpy())}
    final_ids = np.asarray([remap[int(root)] for root in root_ids], dtype=np.int32)
    lengths = pd.Series(final_ids).value_counts().to_numpy()
    diag = {
        "passes": pass_diags,
        "merges": len(all_merges),
        "merge_examples": all_merges[: int(args.max_merge_examples)],
        "final_tracks": int(len(lengths)),
        "final_length_quantiles": quantiles(lengths),
    }
    return final_ids, diag


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {"min": 0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "min": int(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": int(arr.max()),
    }


def write_track_outputs(df: pd.DataFrame, track_ids: np.ndarray, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (out_dir / "trajectory3d").exists():
        shutil.rmtree(out_dir / "trajectory3d")
    traj_dir = out_dir / "trajectory3d"
    traj_dir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["track_id"] = track_ids.astype(np.int32)
    out["laptrack_id"] = out["track_id"]
    tracks_depth = out_dir / "tracks_with_diam_depth.csv"
    tracks_diam = out_dir / "tracks_with_diameter.csv"
    summary_csv = out_dir / "tracks_summary.csv"
    out.to_csv(tracks_depth, index=False)
    drop_cols = [c for c in ("slice", "depth_um", "model_score", "depth_method", "model_crop_method") if c in out.columns]
    out.drop(columns=drop_cols).to_csv(tracks_diam, index=False)

    summary_rows = []
    for tid, g in out.groupby("track_id", sort=True):
        summary_rows.append(
            {
                "track_id": int(tid),
                "len": int(len(g)),
                "frame_first": int(g["frame"].min()),
                "frame_last": int(g["frame"].max()),
            }
        )
        path = traj_dir / f"path_{int(tid):08d}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "xc", "yc", "slice", "diameter_px"])
            writer.writeheader()
            for row in g.sort_values("frame").itertuples(index=False):
                writer.writerow(
                    {
                        "frame": int(row.frame),
                        "xc": float(getattr(row, "seg_xc", getattr(row, "x_lap"))),
                        "yc": float(getattr(row, "seg_yc", getattr(row, "y_lap"))),
                        "slice": float(row.slice),
                        "diameter_px": float(row.diameter_px),
                    }
                )
    summary = pd.DataFrame(summary_rows).sort_values(["len", "track_id"], ascending=[False, True])
    summary.to_csv(summary_csv, index=False)
    lens = summary["len"].to_numpy(np.int32)
    row_count = int(len(out))

    def point_coverage(min_len: int) -> dict[str, int | float]:
        points = int(lens[lens >= min_len].sum()) if len(lens) else 0
        return {
            f"trajectory_points_ge{min_len}": points,
            f"trajectory_point_fraction_ge{min_len}": float(points / row_count) if row_count else 0.0,
        }

    coverage = {}
    for min_len in (2, 3, 5, 10):
        coverage.update(point_coverage(min_len))

    return {
        "tracks_with_diam_depth": str(tracks_depth),
        "tracks_with_diameter": str(tracks_diam),
        "tracks_summary": str(summary_csv),
        "trajectory3d": str(traj_dir),
        "track_count": int(len(summary)),
        "row_count": row_count,
        "one_frame_tracks": int(np.sum(lens == 1)),
        "one_frame_points": int(lens[lens == 1].sum()) if len(lens) else 0,
        **coverage,
        "length_quantiles": quantiles(lens),
    }


def parse_vec3(value: str) -> np.ndarray:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected comma-separated x,y,z values")
    return np.asarray(parts, dtype=np.float64)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage uncertainty-aware LAP linker for holographic particle observations.")
    p.add_argument("--input", required=True, help="Observation CSV created by build_observations.py.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-rho", type=float, default=0.0)
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--z-slices", type=int, default=1024)
    p.add_argument("--p-miss", type=float, default=0.035)
    p.add_argument("--p-longmiss", type=float, default=0.08)
    p.add_argument("--boundary-radius", type=float, default=36.0)
    p.add_argument("--boundary-radius-stage2", type=float, default=72.0)
    p.add_argument("--boundary-softness", type=float, default=18.0)
    p.add_argument("--initial-motion-sigma", type=parse_vec3, default=parse_vec3("5.0,9.0,120.0"))
    p.add_argument("--min-motion-sigma", type=parse_vec3, default=parse_vec3("3.0,5.0,60.0"))
    p.add_argument("--min-velocity-sigma", type=parse_vec3, default=parse_vec3("0.15,0.20,1.2"))
    p.add_argument("--sigma-dv-scale", type=float, default=1.5)
    p.add_argument("--sigma-log-diam-phys", type=float, default=0.18)
    p.add_argument("--embeddings", default="", help="Optional row-aligned ROIFeat embedding .npy.")
    p.add_argument("--feature-scale", type=float, default=0.42)
    p.add_argument("--feature-weight-stage1", type=float, default=0.20)
    p.add_argument("--feature-weight-stage2", type=float, default=0.18)
    p.add_argument("--logdet-weight", type=float, default=0.10)
    p.add_argument("--lambda-rho-stage1", type=float, default=0.04)
    p.add_argument("--diameter-weight-stage1", type=float, default=0.45)
    p.add_argument("--stage1-pos-gate", type=float, default=25.0)
    p.add_argument("--stage1-diam-gate", type=float, default=12.0)
    p.add_argument("--stage1-min-saving", type=float, default=0.0)
    p.add_argument("--stage1-margin", type=float, default=0.06)
    p.add_argument("--no-calibration", action="store_true", help="Use the initial Stage 1 result without self-calibration.")
    p.add_argument("--seed-tracklets-column", default="", help="Use an existing column as Stage 1 tracklets and only run Stage 2 LAP.")
    p.add_argument("--stage2-max-gap", type=int, default=10)
    p.add_argument("--stage2-passes", type=int, default=3)
    p.add_argument("--stage2-solver", choices=["local", "global"], default="local")
    p.add_argument("--stage2-pos-gate", type=float, default=25.0)
    p.add_argument("--stage2-pos-gate-hard", type=float, default=64.0)
    p.add_argument("--stage2-diam-gate", type=float, default=12.0)
    p.add_argument("--stage2-min-saving", type=float, default=0.02)
    p.add_argument("--stage2-gap-cost", type=float, default=0.22)
    p.add_argument("--lambda-quality-stage2", type=float, default=0.14)
    p.add_argument("--pred-weight-stage2", type=float, default=0.55)
    p.add_argument("--velocity-weight-stage2", type=float, default=0.14)
    p.add_argument("--diameter-weight-stage2", type=float, default=0.25)
    p.add_argument("--max-merge-examples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    global EMBEDDINGS
    args = parse_args()
    t0 = time.perf_counter()
    input_path = resolve(args.input)
    out_dir = resolve(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path).reset_index(drop=True)
    if args.embeddings:
        emb_path = resolve(args.embeddings)
        EMBEDDINGS = np.load(emb_path).astype(np.float32)
        norm = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
        EMBEDDINGS = EMBEDDINGS / np.maximum(norm, 1e-8)
        if EMBEDDINGS.shape[0] < len(df):
            raise RuntimeError(f"embedding row count {EMBEDDINGS.shape[0]} is smaller than observation rows {len(df)}")
    else:
        EMBEDDINGS = None
    df["embedding_index"] = df["obs_id"].to_numpy(np.int64)
    if args.start_frame is not None:
        df = df[df["frame"] >= args.start_frame].copy()
    if args.end_frame is not None:
        df = df[df["frame"] <= args.end_frame].copy()
    if args.min_rho > 0:
        df = df[df["rho"] >= args.min_rho].copy()
    df = df.sort_values(["frame", "obs_id"]).reset_index(drop=True)
    df["obs_id"] = np.arange(len(df), dtype=np.int64)
    if df.empty:
        raise RuntimeError("No observations left after filtering")

    flow, flow_diag = estimate_global_flow(df)
    params = LinkParams(
        flow=flow,
        motion_sigma=np.asarray(args.initial_motion_sigma, dtype=np.float64),
        sigma_step=np.asarray(args.initial_motion_sigma, dtype=np.float64),
        sigma_dv=np.maximum(np.asarray(args.initial_motion_sigma, dtype=np.float64) * args.sigma_dv_scale, args.min_velocity_sigma),
    )
    if args.seed_tracklets_column:
        seed_ids = normalize_seed_tracklets(df, args.seed_tracklets_column)
        lengths = pd.Series(seed_ids).value_counts().to_numpy()
        seed_diag = {
            "label": f"seed:{args.seed_tracklets_column}",
            "links": int(len(df) - len(lengths)),
            "tracklets": int(len(lengths)),
            "length_quantiles": quantiles(lengths),
            "counters": {},
            "flow": params.flow.tolist(),
            "motion_sigma": params.motion_sigma.tolist(),
        }
        stage1_initial = Stage1Result(links=[], tracklet_ids=seed_ids, diagnostics=seed_diag)
        stage1 = stage1_initial
        calibration_diag = {"enabled": False, "reason": "seed_tracklets_column"}
    else:
        stage1_initial = run_stage1(df, params, args, "initial")
        if args.no_calibration:
            stage1 = stage1_initial
            calibration_diag = {"enabled": False}
        else:
            params = calibrate_from_links(df, stage1_initial.links, params, args)
            stage1 = run_stage1(df, params, args, "calibrated")
            calibration_diag = {"enabled": True}
    final_ids, stage2_diag = run_stage2(df, stage1.tracklet_ids, params, args)
    output_diag = write_track_outputs(df, final_ids, out_dir, args)

    diag = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "rows": int(len(df)),
        "frames": int(df["frame"].nunique()),
        "global_flow": flow_diag,
        "calibrated_params": {
            "flow": params.flow.tolist(),
            "motion_sigma": params.motion_sigma.tolist(),
            "sigma_step": params.sigma_step.tolist(),
            "sigma_dv": params.sigma_dv.tolist(),
        },
        "stage1_initial": stage1_initial.diagnostics,
        "calibration": calibration_diag,
        "stage1": stage1.diagnostics,
        "stage2": stage2_diag,
        "output": output_diag,
        "args": jsonable(vars(args)),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    (out_dir / "lap_diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
