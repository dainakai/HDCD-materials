from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm


def natural_key(path: str | Path) -> list[Any]:
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoiFeatNet(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DSConv(16, 24, stride=2),
            DSConv(24, 32, stride=2),
            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=2),
            DSConv(64, 96, stride=2),
        )
        self.proj = nn.Sequential(
            nn.Linear(96, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.08),
        )
        self.embedding_head = nn.Linear(96, embedding_dim)
        self.area_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).mean(dim=(2, 3))
        hidden = self.proj(feat)
        emb = F.normalize(self.embedding_head(hidden), dim=1)
        log_area_pred = self.area_head(hidden).squeeze(1)
        return emb, log_area_pred


def fixed_crop_array(arr: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    h, w = arr.shape
    x0 = int(round(cx - crop_size / 2))
    y0 = int(round(cy - crop_size / 2))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return arr[y0:y1, x0:x1]


def focus_center_array(crop: np.ndarray, bw: float, bh: float, crop_size: int, pad_ratio: float) -> np.ndarray:
    pad = max(4.0, pad_ratio * max(bw, bh))
    x0 = max(0, int(round(crop_size / 2 - bw / 2 - pad)))
    y0 = max(0, int(round(crop_size / 2 - bh / 2 - pad)))
    x1 = min(crop_size, int(round(crop_size / 2 + bw / 2 + pad)))
    y1 = min(crop_size, int(round(crop_size / 2 + bh / 2 + pad)))
    out = np.full_like(crop, int(np.median(crop)))
    out[y0:y1, x0:x1] = crop[y0:y1, x0:x1]
    return out


def load_roifeat_model(weights: Path, device: torch.device) -> RoiFeatNet:
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model"]
    embedding_dim = int(state["embedding_head.weight"].shape[0])
    model = RoiFeatNet(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_image_map(image_dir: Path) -> dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return {p.name: p for p in sorted(paths, key=natural_key)}


def compute_roifeat_embeddings(
    df: pd.DataFrame,
    image_dir: Path,
    weights: Path,
    cache_path: Path,
    device_name: str,
    batch_size: int,
    crop_size: int = 64,
    focus_pad_ratio: float = 0.30,
    force: bool = False,
) -> np.ndarray:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if cached.shape[0] == len(df):
            return cached.astype(np.float32, copy=False)

    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model = load_roifeat_model(weights, device)
    image_map = build_image_map(image_dir)
    features: list[np.ndarray] = []
    batch: list[np.ndarray] = []

    with torch.inference_mode():
        for file_name, group in tqdm(df.groupby("file", sort=False), desc="ROIFeat", unit="frame"):
            img_path = image_map.get(str(file_name))
            if img_path is None:
                raise FileNotFoundError(f"Image not found for file={file_name} in {image_dir}")
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            for row in group.itertuples(index=False):
                crop = fixed_crop_array(gray, float(row.xc), float(row.yc), crop_size)
                crop = focus_center_array(crop, float(row.w), float(row.h), crop_size, focus_pad_ratio)
                crop_f = crop.astype(np.float32) / 255.0
                batch.append(crop_f[None, :, :])
                if len(batch) >= batch_size:
                    xb = torch.from_numpy(np.stack(batch, axis=0)).to(device, non_blocking=True)
                    z, _ = model(xb)
                    features.append(z.detach().cpu().numpy().astype(np.float32))
                    batch = []
        if batch:
            xb = torch.from_numpy(np.stack(batch, axis=0)).to(device, non_blocking=True)
            z, _ = model(xb)
            features.append(z.detach().cpu().numpy().astype(np.float32))

    emb = np.concatenate(features, axis=0)
    emb = l2_normalize(emb.astype(np.float32))
    ensure_dir(cache_path.parent)
    np.save(cache_path, emb)
    return emb


def estimate_global_flow(
    df: pd.DataFrame,
    sample_pairs: int = 80,
    min_conf: float = 0.40,
    radius: float = 12.0,
) -> tuple[float, float, dict[str, Any]]:
    frames = sorted(df["frame"].unique())
    grid = [(dx, dy) for dx in np.linspace(-2.5, 2.5, 21) for dy in np.linspace(1.0, 8.0, 29)]
    shifts: list[tuple[float, float]] = []

    for f0, f1 in zip(frames[:sample_pairs], frames[1:sample_pairs + 1]):
        a = df[(df.frame == f0) & (df.conf >= min_conf)][["xc", "yc"]].to_numpy(np.float32)
        b = df[(df.frame == f1) & (df.conf >= min_conf)][["xc", "yc"]].to_numpy(np.float32)
        if len(a) < 50 or len(b) < 50:
            continue
        sample = a[:: max(1, len(a) // 260)]
        tree = cKDTree(b)
        best: tuple[float, float, float] | None = None
        for dx, dy in grid:
            d, _idx = tree.query(sample + np.array([dx, dy], dtype=np.float32), k=1, distance_upper_bound=radius)
            finite = d[np.isfinite(d)]
            if len(finite) < len(sample) * 0.65:
                continue
            score = float(np.median(finite) + 0.20 * np.percentile(finite, 90))
            if best is None or score < best[0]:
                best = (score, float(dx), float(dy))
        if best is not None:
            shifts.append((best[1], best[2]))

    if not shifts:
        return -0.75, 4.0, {"estimated": False, "shifts": []}
    arr = np.asarray(shifts, dtype=np.float32)
    dx, dy = np.median(arr, axis=0).tolist()
    return float(dx), float(dy), {
        "estimated": True,
        "sample_pairs": len(shifts),
        "median_dx": float(dx),
        "median_dy": float(dy),
        "mean_dx": float(arr[:, 0].mean()),
        "mean_dy": float(arr[:, 1].mean()),
        "q_dx": np.quantile(arr[:, 0], [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "q_dy": np.quantile(arr[:, 1], [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


@dataclass
class TrackState:
    track_id: int
    last_idx: int
    last_frame: int
    pos: np.ndarray
    velocity: np.ndarray
    size_log: float
    emb: np.ndarray
    hits: int = 1
    age: int = 1


def track_gate(dt: int, area_sqrt: float, speed: float, base_gate: float, gap_gate: float) -> float:
    return float(min(42.0, max(7.0, base_gate + gap_gate * max(0, dt - 1) + 0.22 * speed + 0.18 * area_sqrt)))


def online_link(
    df: pd.DataFrame,
    emb: np.ndarray | None,
    global_flow: tuple[float, float],
    *,
    max_gap: int,
    base_gate: float,
    gap_gate: float,
    startup_gate: float,
    max_cost: float,
    motion_weight: float,
    feature_weight: float,
    size_weight: float,
    age_bonus: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(df)
    assigned = np.zeros(n, dtype=np.int32)
    tracks: dict[int, TrackState] = {}
    active: set[int] = set()
    next_id = 1
    frames = sorted(df["frame"].unique())
    pos_all = df[["xc", "yc"]].to_numpy(np.float32)
    area_sqrt_all = np.sqrt(np.maximum(1.0, (df["w"].to_numpy(np.float32) * df["h"].to_numpy(np.float32))))
    size_log_all = np.log(np.maximum(1.0, df["w"].to_numpy(np.float32) * df["h"].to_numpy(np.float32)))
    frame_indices = {frame: grp.index.to_numpy(np.int64) for frame, grp in df.groupby("frame", sort=True)}
    gf = np.asarray(global_flow, dtype=np.float32)

    counters = defaultdict(int)
    for frame in tqdm(frames, desc="Online link", unit="frame"):
        det_idx = frame_indices[frame]
        det_pos = pos_all[det_idx]
        det_size_log = size_log_all[det_idx]

        live_ids = [tid for tid in active if 0 < frame - tracks[tid].last_frame <= max_gap]
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if live_ids and len(det_idx):
            pred_pos = np.empty((len(live_ids), 2), dtype=np.float32)
            gates = np.empty(len(live_ids), dtype=np.float32)
            pred_vels = np.empty((len(live_ids), 2), dtype=np.float32)
            for i, tid in enumerate(live_ids):
                st = tracks[tid]
                dt = frame - st.last_frame
                if st.hits < 2:
                    v = gf
                else:
                    v = 0.88 * st.velocity + 0.12 * gf
                speed = float(np.linalg.norm(v))
                pred_vels[i] = v
                pred_pos[i] = st.pos + v * dt
                gates[i] = track_gate(dt, math.exp(0.5 * st.size_log), speed, base_gate, gap_gate)
                if st.hits < 2 and dt == 1:
                    gates[i] = max(gates[i], startup_gate)

            diff = pred_pos[:, None, :] - det_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            finite = dist <= gates[:, None]
            cost = np.full(dist.shape, 1e6, dtype=np.float32)
            if np.any(finite):
                motion_cost = (dist / np.maximum(gates[:, None], 1e-6)) ** 2
                size_cost = np.abs(np.asarray([tracks[tid].size_log for tid in live_ids], dtype=np.float32)[:, None] - det_size_log[None, :])
                size_cost = np.clip(size_cost / 1.25, 0.0, 2.0)
                cost_part = motion_weight * motion_cost + size_weight * size_cost
                if age_bonus > 0:
                    age = np.asarray([tracks[tid].hits for tid in live_ids], dtype=np.float32)
                    bonus = age_bonus * np.minimum(1.0, np.log1p(age) / math.log(40.0))
                    cost_part -= bonus[:, None]
                if emb is not None:
                    track_emb = np.stack([tracks[tid].emb for tid in live_ids], axis=0)
                    feat_dist = 1.0 - np.clip(track_emb @ emb[det_idx].T, -1.0, 1.0)
                    cost_part += feature_weight * np.clip(feat_dist / 0.42, 0.0, 2.5)
                cost[finite] = cost_part[finite]
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] > max_cost or not finite[r, c]:
                        continue
                    tid = live_ids[int(r)]
                    j = int(det_idx[int(c)])
                    st = tracks[tid]
                    dt = frame - st.last_frame
                    measured_v = (pos_all[j] - st.pos) / max(1, dt)
                    if st.hits < 2:
                        new_v = 0.15 * gf + 0.85 * measured_v
                    else:
                        new_v = 0.72 * st.velocity + 0.28 * measured_v
                    st.velocity = new_v.astype(np.float32)
                    st.pos = pos_all[j].astype(np.float32)
                    st.last_idx = j
                    st.last_frame = int(frame)
                    st.size_log = 0.86 * st.size_log + 0.14 * float(size_log_all[j])
                    if emb is not None:
                        st.emb = l2_normalize((0.88 * st.emb + 0.12 * emb[j])[None, :])[0].astype(np.float32)
                    st.hits += 1
                    st.age += dt
                    assigned[j] = tid
                    matched_tracks.add(tid)
                    matched_dets.add(j)
                    counters["online_matches"] += 1
                    if dt > 1:
                        counters["online_gap_matches"] += 1

        for j in det_idx:
            if int(j) in matched_dets:
                continue
            tid = next_id
            next_id += 1
            e = emb[j].astype(np.float32) if emb is not None else np.zeros(1, dtype=np.float32)
            tracks[tid] = TrackState(
                track_id=tid,
                last_idx=int(j),
                last_frame=int(frame),
                pos=pos_all[j].astype(np.float32),
                velocity=gf.copy(),
                size_log=float(size_log_all[j]),
                emb=e,
            )
            assigned[j] = tid
            matched_tracks.add(tid)
            active.add(tid)
            counters["new_tracks"] += 1

        stale = [tid for tid in active if frame - tracks[tid].last_frame > max_gap]
        for tid in stale:
            active.remove(tid)

    diag = {
        "initial_track_count": int(next_id - 1),
        "online_matches": int(counters["online_matches"]),
        "online_gap_matches": int(counters["online_gap_matches"]),
        "new_tracks": int(counters["new_tracks"]),
    }
    return assigned, diag


class UnionFind:
    def __init__(self, values: np.ndarray) -> None:
        self.parent = {int(v): int(v) for v in values}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        root = min(ra, rb)
        other = max(ra, rb)
        self.parent[other] = root
        return True


def tracklet_summary(df: pd.DataFrame, track_ids: np.ndarray, emb: np.ndarray | None) -> dict[int, dict[str, Any]]:
    tmp = pd.DataFrame(
        {
            "track_id": track_ids,
            "frame": df["frame"].to_numpy(np.int32),
            "xc": df["xc"].to_numpy(np.float32),
            "yc": df["yc"].to_numpy(np.float32),
            "area": np.maximum(1.0, df["w"].to_numpy(np.float32) * df["h"].to_numpy(np.float32)),
            "idx": np.arange(len(df), dtype=np.int32),
        }
    )
    out: dict[int, dict[str, Any]] = {}
    for tid, g in tmp.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        idx = g["idx"].to_numpy(np.int32)
        frames = g["frame"].to_numpy(np.int32)
        pos = g[["xc", "yc"]].to_numpy(np.float32)
        area_log = np.log(g["area"].to_numpy(np.float32))

        def endpoint_velocity(tail: bool) -> np.ndarray:
            take = min(6, len(g))
            pp = pos[-take:] if tail else pos[:take]
            ff = frames[-take:] if tail else frames[:take]
            if len(pp) >= 2 and ff[-1] != ff[0]:
                dt = (ff - ff.mean()).astype(np.float32)
                denom = float(np.sum(dt * dt))
                if denom > 0:
                    vx = float(np.sum(dt * (pp[:, 0] - pp[:, 0].mean())) / denom)
                    vy = float(np.sum(dt * (pp[:, 1] - pp[:, 1].mean())) / denom)
                    return np.array([vx, vy], dtype=np.float32)
            return np.zeros(2, dtype=np.float32)

        if emb is not None:
            head = l2_normalize(emb[idx[: min(5, len(idx))]].mean(axis=0, keepdims=True))[0]
            tail = l2_normalize(emb[idx[-min(5, len(idx)) :]].mean(axis=0, keepdims=True))[0]
        else:
            head = tail = np.zeros(1, dtype=np.float32)

        out[int(tid)] = {
            "id": int(tid),
            "len": int(len(g)),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_pos": pos[0],
            "end_pos": pos[-1],
            "start_v": endpoint_velocity(tail=False),
            "end_v": endpoint_velocity(tail=True),
            "start_size": float(np.median(area_log[: min(5, len(area_log))])),
            "end_size": float(np.median(area_log[-min(5, len(area_log)) :])),
            "head_emb": head.astype(np.float32),
            "tail_emb": tail.astype(np.float32),
        }
    return out


def stitch_tracklets(
    df: pd.DataFrame,
    track_ids: np.ndarray,
    emb: np.ndarray | None,
    global_flow: tuple[float, float],
    *,
    max_gap: int,
    max_cost: float,
    base_gate: float,
    gap_gate: float,
    startup_gate: float,
    min_len_for_feature: int,
    large_sqrt_area: float,
    large_gate_scale: float,
    large_max_cost: float,
    large_feature_dist: float,
    large_size_log_max: float,
    large_vel_cost_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_gap <= 0:
        return track_ids, {"stitch_merges": 0}

    unique_ids = np.unique(track_ids)
    uf = UnionFind(unique_ids)
    summaries = tracklet_summary(df, track_ids, emb)
    gf = np.asarray(global_flow, dtype=np.float32)
    merges: list[dict[str, Any]] = []
    counters = defaultdict(int)

    for _pass in range(2):
        roots = {tid: uf.find(tid) for tid in summaries}
        root_members: dict[int, list[int]] = defaultdict(list)
        for tid, root in roots.items():
            root_members[root].append(tid)

        root_summary: dict[int, dict[str, Any]] = {}
        for root, members in root_members.items():
            # The original fragments are non-overlapping because we only union end-before-start pairs.
            starts = [summaries[m] for m in members]
            first = min(starts, key=lambda s: s["start_frame"])
            last = max(starts, key=lambda s: s["end_frame"])
            root_summary[root] = {
                "id": root,
                "len": int(sum(s["len"] for s in starts)),
                "start_frame": first["start_frame"],
                "end_frame": last["end_frame"],
                "start_pos": first["start_pos"],
                "end_pos": last["end_pos"],
                "start_v": first["start_v"],
                "end_v": last["end_v"],
                "start_size": first["start_size"],
                "end_size": last["end_size"],
                "head_emb": first["head_emb"],
                "tail_emb": last["tail_emb"],
            }

        by_start: dict[int, list[int]] = defaultdict(list)
        for tid, s in root_summary.items():
            by_start[int(s["start_frame"])].append(tid)

        pass_merges = 0
        used_starts: set[int] = set()
        used_ends: set[int] = set()
        for gap in range(1, max_gap + 1):
            ends = [tid for tid, s in root_summary.items() if tid not in used_ends]
            starts_by_frame = {f: [tid for tid in ids if tid not in used_starts] for f, ids in by_start.items()}
            pairs: list[tuple[Any, ...]] = []
            for a in ends:
                sa = root_summary[a]
                target_frame = int(sa["end_frame"]) + gap
                starts = starts_by_frame.get(target_frame)
                if not starts:
                    continue
                v = sa["end_v"]
                if np.linalg.norm(v) < 0.5:
                    v = gf
                pred = sa["end_pos"] + v * gap
                b_pos = np.stack([root_summary[b]["start_pos"] for b in starts], axis=0)
                dist = np.linalg.norm(b_pos - pred[None, :], axis=1)
                speed = float(np.linalg.norm(v))
                gate = track_gate(gap, math.exp(0.5 * sa["end_size"]), speed, base_gate, gap_gate)
                if sa["len"] < 2 and gap == 1:
                    gate = max(gate, startup_gate)
                for j, b in enumerate(starts):
                    sb = root_summary[b]
                    end_sqrt_area = math.exp(0.5 * sa["end_size"])
                    start_sqrt_area = math.exp(0.5 * sb["start_size"])
                    large_pair = min(end_sqrt_area, start_sqrt_area) >= large_sqrt_area
                    effective_gate = gate * large_gate_scale if large_pair else gate
                    if dist[j] > effective_gate:
                        continue

                    if large_pair:
                        counters["large_candidates"] += 1

                    size_log_delta = abs(sa["end_size"] - sb["start_size"])
                    size_cost = min(2.0, size_log_delta / 1.25)
                    vel_b = sb["start_v"]
                    if np.linalg.norm(vel_b) < 0.5:
                        vel_b = gf
                    vel_cost = min(2.0, float(np.linalg.norm(v - vel_b)) / 6.0)
                    feat_cost = 0.0
                    feature_dist = None
                    use_feature = emb is not None and (
                        sa["len"] >= min_len_for_feature or sb["len"] >= min_len_for_feature or large_pair
                    )
                    if use_feature:
                        feat_dist = 1.0 - float(np.clip(np.dot(sa["tail_emb"], sb["head_emb"]), -1.0, 1.0))
                        feature_dist = feat_dist
                        feat_cost = min(2.5, feat_dist / 0.42)

                    normal_cost = (
                        0.55 * (float(dist[j]) / max(gate, 1e-6)) ** 2
                        + 0.15 * vel_cost
                        + 0.12 * size_cost
                        + 0.18 * feat_cost
                    )
                    cost = normal_cost
                    effective_max_cost = max_cost
                    accepted_by = "normal"
                    large_feature_ok = (
                        large_pair
                        and feature_dist is not None
                        and feature_dist <= large_feature_dist
                        and size_log_delta <= large_size_log_max
                        and vel_cost <= large_vel_cost_max
                    )
                    if large_feature_ok:
                        counters["large_feature_ok_candidates"] += 1
                        large_cost = (
                            0.48 * (float(dist[j]) / max(effective_gate, 1e-6)) ** 2
                            + 0.12 * vel_cost
                            + 0.08 * size_cost
                            + 0.12 * feat_cost
                        )
                        if large_cost < cost:
                            cost = large_cost
                        effective_max_cost = max(max_cost, large_max_cost)
                        if normal_cost > max_cost:
                            accepted_by = "large_feature"

                    if cost <= effective_max_cost:
                        pairs.append(
                            (
                                cost,
                                a,
                                b,
                                float(dist[j]),
                                float(effective_gate),
                                float(feat_cost),
                                float(feature_dist) if feature_dist is not None else None,
                                bool(large_feature_ok),
                                float(end_sqrt_area),
                                float(start_sqrt_area),
                                accepted_by,
                                float(normal_cost),
                            )
                        )
            for (
                cost,
                a,
                b,
                dist,
                gate,
                feat_cost,
                feature_dist,
                large_feature_ok,
                end_sqrt_area,
                start_sqrt_area,
                accepted_by,
                normal_cost,
            ) in sorted(pairs, key=lambda x: x[0]):
                if a in used_ends or b in used_starts:
                    continue
                if uf.find(a) == uf.find(b):
                    continue
                if uf.union(a, b):
                    used_ends.add(a)
                    used_starts.add(b)
                    pass_merges += 1
                    merges.append(
                        {
                            "pass": _pass,
                            "gap": gap,
                            "from": int(a),
                            "to": int(b),
                            "cost": float(cost),
                            "dist": dist,
                            "gate": gate,
                            "feature_cost": feat_cost,
                            "feature_dist": feature_dist,
                            "large_feature": large_feature_ok,
                            "end_sqrt_area": end_sqrt_area,
                            "start_sqrt_area": start_sqrt_area,
                            "accepted_by": accepted_by,
                            "normal_cost": normal_cost,
                        }
                    )
                    if large_feature_ok:
                        counters["large_feature_merges"] += 1
                        if accepted_by == "large_feature":
                            counters["large_relaxed_merges"] += 1
                        else:
                            counters["large_normal_merges"] += 1
        if pass_merges == 0:
            break

    root_ids = np.asarray([uf.find(int(t)) for t in track_ids], dtype=np.int32)
    roots_sorted = sorted(np.unique(root_ids), key=lambda t: (df.loc[np.where(root_ids == t)[0][0], "frame"], t))
    remap = {old: i + 1 for i, old in enumerate(roots_sorted)}
    stitched = np.asarray([remap[int(t)] for t in root_ids], dtype=np.int32)
    return stitched, {
        "stitch_merges": len(merges),
        "large_candidates": int(counters["large_candidates"]),
        "large_feature_ok_candidates": int(counters["large_feature_ok_candidates"]),
        "large_feature_merges": int(counters["large_feature_merges"]),
        "large_relaxed_merges": int(counters["large_relaxed_merges"]),
        "large_normal_merges": int(counters["large_normal_merges"]),
        "large_params": {
            "sqrt_area": float(large_sqrt_area),
            "gate_scale": float(large_gate_scale),
            "max_cost": float(large_max_cost),
            "feature_dist": float(large_feature_dist),
            "size_log_max": float(large_size_log_max),
            "vel_cost_max": float(large_vel_cost_max),
        },
        "stitch_examples": merges[:50],
    }


CONTRAST_AREA_FRACTION = 0.70
CONTRAST_AREA_BG_PERCENTILE = 90.0
CONTRAST_AREA_PEAK_PERCENTILE = 95.0
CONTRAST_AREA_CORE_RADIUS_SCALE = 0.30
CONTRAST_AREA_ROI_SCALE = 1.8
CONTRAST_AREA_ROI_MARGIN_PX = 4.0
CONTRAST_AREA_MIN_ROI_RADIUS_PX = 7.0
CONTRAST_AREA_FULL_GATE_PX = 2.0


def component_for_particle(
    binary: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    max_area_fraction: float = 0.70,
):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if nlabels <= 1:
        return None

    cx = int(round(center_x))
    cy = int(round(center_y))
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        label = int(labels[cy, cx])
        if label > 0:
            area = float(stats[label, cv2.CC_STAT_AREA])
            if 2 <= area <= binary.size * max_area_fraction:
                return label, stats[label], centroids[label]

    best = None
    best_score = None
    roi_area = binary.shape[0] * binary.shape[1]
    for label in range(1, nlabels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > roi_area * max_area_fraction:
            continue
        mx, my = centroids[label]
        dist2 = (mx - center_x) ** 2 + (my - center_y) ** 2
        score = dist2 / max(area, 1.0)
        if best_score is None or score < best_score:
            best = (label, stats[label], centroids[label])
            best_score = score
    return best


def measure_particle(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float, xc: float, yc: float):
    h, w = gray.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    bbox_diameter_px = max(bw, bh)
    radius = max(
        CONTRAST_AREA_MIN_ROI_RADIUS_PX,
        0.5 * bbox_diameter_px * CONTRAST_AREA_ROI_SCALE + CONTRAST_AREA_ROI_MARGIN_PX,
    )
    crop_radius = int(math.ceil(radius))

    rx1 = max(0, int(math.floor(xc)) - crop_radius)
    ry1 = max(0, int(math.floor(yc)) - crop_radius)
    rx2 = min(w, int(math.floor(xc)) + crop_radius + 1)
    ry2 = min(h, int(math.floor(yc)) + crop_radius + 1)

    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_empty_roi", np.nan, 0

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    local_cx = xc - rx1
    local_cy = yc - ry1

    yy, xx = np.indices(blur.shape)
    rr = np.hypot(xx - local_cx, yy - local_cy)
    background = float(np.percentile(blur, CONTRAST_AREA_BG_PERCENTILE))
    darkness = np.clip(background - blur.astype(np.float32), 0.0, None)
    core = darkness[rr <= max(2.0, bbox_diameter_px * CONTRAST_AREA_CORE_RADIUS_SCALE)]
    if core.size == 0:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_core", np.nan, 0

    peak = float(np.percentile(core, CONTRAST_AREA_PEAK_PERCENTILE))
    if peak <= 1e-6:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_contrast", np.nan, 0

    binary = (darkness >= CONTRAST_AREA_FRACTION * peak).astype(np.uint8) * 255
    selected = component_for_particle(binary, local_cx, local_cy)

    if selected is None:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_component", np.nan, 0

    _label, stats, centroid = selected
    area = float(stats[cv2.CC_STAT_AREA])
    diameter_px = 2.0 * math.sqrt(area / math.pi)
    seg_x = rx1 + float(centroid[0])
    seg_y = ry1 + float(centroid[1])
    if math.hypot(seg_x - xc, seg_y - yc) > CONTRAST_AREA_FULL_GATE_PX:
        return (
            bbox_diameter_px,
            xc,
            yc,
            rx1,
            ry1,
            rx2,
            ry2,
            "contrast_area_f70_bg90_full_gate2px_bbox_fallback",
            np.nan,
            0,
        )
    return diameter_px, seg_x, seg_y, rx1, ry1, rx2, ry2, "contrast_area_f70_bg90_component", np.nan, int(area)


def color_from_id(track_id: int) -> tuple[int, int, int]:
    x = (track_id * 123457) & 0xFFFFFFFF
    b = 50 + (x % 206)
    g = 50 + ((x >> 8) % 206)
    r = 50 + ((x >> 16) % 206)
    return int(b), int(g), int(r)


def write_outputs(
    df: pd.DataFrame,
    track_ids: np.ndarray,
    image_dir: Path,
    out_dir: Path,
    *,
    pixel_pitch_um: float,
    write_per_track: bool,
    vis_stride: int,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    tracks_csv_dir = ensure_dir(out_dir / "tracks_csv") if write_per_track else None
    vis_dir = ensure_dir(out_dir / "vis") if vis_stride > 0 else None
    image_map = build_image_map(image_dir)

    fieldnames = [
        "frame", "file",
        "track_id", "class_id", "conf",
        "x1", "y1", "x2", "y2",
        "xc", "yc", "w", "h",
        "x1n", "y1n", "x2n", "y2n",
        "xcn", "ycn", "wn", "hn",
        "img_w", "img_h",
        "row_id",
        "diameter_px", "diameter_um",
        "seg_xc", "seg_yc",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "diameter_method", "otsu_threshold", "seg_area_px",
    ]
    all_path = out_dir / "tracks_all.csv"
    diam_path = out_dir / "tracks_with_diameter.csv"
    summary_path = out_dir / "tracks_summary.csv"
    rows: list[dict[str, Any]] = []
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"len": 0, "frame_first": 0, "frame_last": 0})
    per_track_handles: dict[int, Any] = {}
    per_track_writers: dict[int, csv.DictWriter] = {}

    with all_path.open("w", newline="", encoding="utf-8") as all_f, diam_path.open("w", newline="", encoding="utf-8") as diam_f:
        all_w = csv.DictWriter(all_f, fieldnames=fieldnames)
        diam_w = csv.DictWriter(diam_f, fieldnames=fieldnames)
        all_w.writeheader()
        diam_w.writeheader()
        row_id = 0
        for file_name, group in tqdm(df.assign(track_id=track_ids).groupby("file", sort=False), desc="Write CSV", unit="frame"):
            img_path = image_map.get(str(file_name))
            if img_path is None:
                raise FileNotFoundError(f"Image not found for file={file_name} in {image_dir}")
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            frame = int(group["frame"].iloc[0])
            vis_img = None
            if vis_dir is not None and (frame % vis_stride == 0):
                vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            for row in group.itertuples(index=False):
                diameter_px, seg_x, seg_y, rx1, ry1, rx2, ry2, method, otsu_threshold, area = measure_particle(
                    gray, float(row.x1), float(row.y1), float(row.x2), float(row.y2), float(row.xc), float(row.yc)
                )
                img_w = int(row.img_w)
                img_h = int(row.img_h)
                out = {
                    "frame": int(row.frame),
                    "file": str(row.file),
                    "track_id": int(row.track_id),
                    "class_id": int(row.class_id),
                    "conf": float(row.conf),
                    "x1": float(row.x1), "y1": float(row.y1), "x2": float(row.x2), "y2": float(row.y2),
                    "xc": float(row.xc), "yc": float(row.yc), "w": float(row.w), "h": float(row.h),
                    "x1n": float(row.x1 / img_w), "y1n": float(row.y1 / img_h),
                    "x2n": float(row.x2 / img_w), "y2n": float(row.y2 / img_h),
                    "xcn": float(row.xc / img_w), "ycn": float(row.yc / img_h),
                    "wn": float(row.w / img_w), "hn": float(row.h / img_h),
                    "img_w": img_w,
                    "img_h": img_h,
                    "row_id": row_id,
                    "diameter_px": f"{diameter_px:.6f}",
                    "diameter_um": f"{diameter_px * pixel_pitch_um:.6f}",
                    "seg_xc": f"{seg_x:.6f}",
                    "seg_yc": f"{seg_y:.6f}",
                    "roi_x1": int(rx1),
                    "roi_y1": int(ry1),
                    "roi_x2": int(rx2),
                    "roi_y2": int(ry2),
                    "diameter_method": method,
                    "otsu_threshold": f"{otsu_threshold:.6f}" if np.isfinite(otsu_threshold) else "",
                    "seg_area_px": int(area),
                }
                row_id += 1
                all_w.writerow(out)
                diam_w.writerow(out)
                rows.append(out)

                tid = int(row.track_id)
                st = stats[tid]
                st["len"] += 1
                if st["len"] == 1:
                    st["frame_first"] = int(row.frame)
                st["frame_last"] = int(row.frame)

                if write_per_track and tracks_csv_dir is not None:
                    if tid not in per_track_writers:
                        f = (tracks_csv_dir / f"track_{tid:010d}.csv").open("w", newline="", encoding="utf-8")
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        per_track_handles[tid] = f
                        per_track_writers[tid] = w
                    per_track_writers[tid].writerow(out)

                if vis_img is not None:
                    color = color_from_id(tid)
                    x1 = int(round(float(row.x1)))
                    y1 = int(round(float(row.y1)))
                    x2 = int(round(float(row.x2)))
                    y2 = int(round(float(row.y2)))
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_img, str(tid), (x1, max(0, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            if vis_img is not None:
                cv2.imwrite(str(vis_dir / str(file_name)), vis_img)

    for f in per_track_handles.values():
        f.close()

    summary_rows = [
        {"track_id": tid, "len": st["len"], "frame_first": st["frame_first"], "frame_last": st["frame_last"]}
        for tid, st in stats.items()
    ]
    summary_rows.sort(key=lambda r: (-r["len"], r["track_id"]))
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "len", "frame_first", "frame_last"])
        writer.writeheader()
        writer.writerows(summary_rows)

    lens = np.asarray([r["len"] for r in summary_rows], dtype=np.int32)
    row_count = int(len(df))

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
        "tracks_all": str(all_path),
        "tracks_with_diameter": str(diam_path),
        "tracks_summary": str(summary_path),
        "track_count": int(len(summary_rows)),
        "row_count": row_count,
        "length_quantiles": {
            "min": int(lens.min()) if len(lens) else 0,
            "p25": float(np.quantile(lens, 0.25)) if len(lens) else 0,
            "median": float(np.quantile(lens, 0.50)) if len(lens) else 0,
            "p75": float(np.quantile(lens, 0.75)) if len(lens) else 0,
            "p90": float(np.quantile(lens, 0.90)) if len(lens) else 0,
            "p95": float(np.quantile(lens, 0.95)) if len(lens) else 0,
            "p99": float(np.quantile(lens, 0.99)) if len(lens) else 0,
            "max": int(lens.max()) if len(lens) else 0,
        },
        "one_frame_tracks": int(np.sum(lens == 1)) if len(lens) else 0,
        "one_frame_points": int(lens[lens == 1].sum()) if len(lens) else 0,
        **coverage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hologram-specific 2D linker for YOLO raw detections.")
    parser.add_argument("--raw-csv", default="runs/C000H001S0011/raw_detections.csv")
    parser.add_argument("--image-dir", default="260106/05_03_xyproj/C000H001S0011")
    parser.add_argument("--out-dir", default="runs/C000H001S0011")
    parser.add_argument("--roifeat-weights", default="models/roi/roifeat_best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-conf", type=float, default=0.10)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-roifeat", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--online-max-gap", type=int, default=5)
    parser.add_argument("--stitch-max-gap", type=int, default=12)
    parser.add_argument("--base-gate", type=float, default=8.0)
    parser.add_argument("--gap-gate", type=float, default=3.2)
    parser.add_argument("--startup-gate", type=float, default=24.0)
    parser.add_argument("--online-max-cost", type=float, default=0.90)
    parser.add_argument("--stitch-max-cost", type=float, default=0.62)
    parser.add_argument("--motion-weight", type=float, default=0.95)
    parser.add_argument("--feature-weight", type=float, default=0.24)
    parser.add_argument("--size-weight", type=float, default=0.18)
    parser.add_argument("--age-bonus", type=float, default=0.05)
    parser.add_argument("--large-stitch-sqrt-area", type=float, default=9.0)
    parser.add_argument("--large-stitch-gate-scale", type=float, default=1.65)
    parser.add_argument("--large-stitch-max-cost", type=float, default=1.02)
    parser.add_argument("--large-stitch-feature-dist", type=float, default=0.58)
    parser.add_argument("--large-stitch-size-log-max", type=float, default=0.85)
    parser.add_argument("--large-stitch-vel-cost-max", type=float, default=1.35)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--per-track-csv", action="store_true")
    parser.add_argument("--vis-stride", type=int, default=0, help="Write visualization for frames divisible by this value. 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    raw_path = Path(args.raw_csv)
    image_dir = Path(args.image_dir)
    df = pd.read_csv(raw_path)
    if args.start_frame is not None:
        df = df[df["frame"] >= args.start_frame]
    if args.end_frame is not None:
        df = df[df["frame"] <= args.end_frame]
    df = df[df["conf"] >= args.min_conf].copy()
    if "raw_det_id" in df.columns:
        order_col = "raw_det_id"
    elif "raw_row_id" in df.columns:
        order_col = "raw_row_id"
    elif "row_id" in df.columns:
        order_col = "row_id"
    else:
        df["_input_order"] = np.arange(len(df), dtype=np.int64)
        order_col = "_input_order"
    df = df.sort_values(["frame", order_col]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No detections left after filtering.")

    print(f"Detections: {len(df)} rows, frames {df.frame.min()}..{df.frame.max()}, files {df.file.nunique()}")
    print(f"Device requested for ROIFeat: {args.device}")

    dx, dy, flow_diag = estimate_global_flow(df)
    print(f"Estimated global flow: dx={dx:.3f}, dy={dy:.3f}")

    emb = None
    if not args.no_roifeat:
        weights = Path(args.roifeat_weights)
        cache_path = out_dir / "roifeat_embeddings.npy"
        emb = compute_roifeat_embeddings(
            df,
            image_dir,
            weights,
            cache_path,
            args.device,
            args.feature_batch_size,
            force=args.force_features,
        )
        print(f"ROIFeat embeddings: {emb.shape}")

    initial_ids, online_diag = online_link(
        df,
        emb,
        (dx, dy),
        max_gap=args.online_max_gap,
        base_gate=args.base_gate,
        gap_gate=args.gap_gate,
        startup_gate=args.startup_gate,
        max_cost=args.online_max_cost,
        motion_weight=args.motion_weight,
        feature_weight=0.0 if emb is None else args.feature_weight,
        size_weight=args.size_weight,
        age_bonus=args.age_bonus,
    )
    stitched_ids, stitch_diag = stitch_tracklets(
        df,
        initial_ids,
        emb,
        (dx, dy),
        max_gap=args.stitch_max_gap,
        max_cost=args.stitch_max_cost,
        base_gate=args.base_gate,
        gap_gate=args.gap_gate,
        startup_gate=args.startup_gate,
        min_len_for_feature=2,
        large_sqrt_area=args.large_stitch_sqrt_area,
        large_gate_scale=args.large_stitch_gate_scale,
        large_max_cost=args.large_stitch_max_cost,
        large_feature_dist=args.large_stitch_feature_dist,
        large_size_log_max=args.large_stitch_size_log_max,
        large_vel_cost_max=args.large_stitch_vel_cost_max,
    )
    out_diag = write_outputs(
        df,
        stitched_ids,
        image_dir,
        out_dir,
        pixel_pitch_um=args.pixel_pitch_um,
        write_per_track=args.per_track_csv,
        vis_stride=args.vis_stride,
    )

    diag = {
        "args": vars(args),
        "raw_csv": str(raw_path),
        "image_dir": str(image_dir),
        "detections": {
            "rows": int(len(df)),
            "frame_min": int(df["frame"].min()),
            "frame_max": int(df["frame"].max()),
            "frame_count": int(df["frame"].nunique()),
            "rows_per_frame_median": float(df.groupby("frame").size().median()),
            "rows_per_frame_mean": float(df.groupby("frame").size().mean()),
        },
        "global_flow": flow_diag,
        "online": online_diag,
        "stitch": stitch_diag,
        "output": out_diag,
        "elapsed_sec": float(time.perf_counter() - t_start),
    }
    with (out_dir / "link_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps({k: diag[k] for k in ["detections", "online", "stitch", "output", "elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
