"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
FRAMES=6000
TRUTH_PER_FRAME=200
PIXEL_UM=10.0
SLICE_UM=100.0
XY_GATE_PX=8.0
TRUTH_PATH=LOW_CONF_DETECTIONS=Path('.')

def load_truth() -> dict[str, np.ndarray]:
    print("Loading truth labels...", flush=True)
    frame = pd.read_parquet(TRUTH_PATH)
    frame = (
        frame.loc[frame["center_in_volume"].eq(1)]
        .sort_values(["frame", "particle_id"], kind="stable")
        .reset_index(drop=True)
    )
    counts = frame.groupby("frame", sort=True).size().to_numpy()
    if len(frame) != FRAMES * TRUTH_PER_FRAME or not np.all(counts == TRUTH_PER_FRAME):
        raise RuntimeError(
            f"Truth contract changed: rows={len(frame)}, per-frame={np.unique(counts)}"
        )
    if not np.array_equal(frame["frame"].to_numpy(), np.repeat(np.arange(FRAMES), TRUTH_PER_FRAME)):
        raise RuntimeError("Truth frames are not the expected contiguous 0..5999 blocks")
    return {
        "id": frame["particle_id"].to_numpy().reshape(FRAMES, TRUTH_PER_FRAME),
        "x": frame["x_px"].to_numpy().reshape(FRAMES, TRUTH_PER_FRAME),
        "y": frame["y_px"].to_numpy().reshape(FRAMES, TRUTH_PER_FRAME),
        "z": frame["z_slice"].to_numpy().reshape(FRAMES, TRUTH_PER_FRAME),
        "diameter": frame["diameter_um"].to_numpy().reshape(FRAMES, TRUTH_PER_FRAME),
    }

def greedy_detection_flags(
    truth: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    print("Loading low-confidence detections...", flush=True)
    detection = pd.read_parquet(LOW_CONF_DETECTIONS)
    detection = detection.sort_values(["frame", "conf"], ascending=[True, False], kind="stable").reset_index(drop=True)
    frames = detection["frame"].to_numpy(np.int32)
    confidence = detection["conf"].to_numpy(np.float64)
    pred_x = detection["xc"].to_numpy(np.float64)
    pred_y = detection["yc"].to_numpy(np.float64)
    counts = np.bincount(frames, minlength=FRAMES)
    bounds = np.r_[0, np.cumsum(counts)]
    flags = np.zeros(len(detection), dtype=bool)

    print("Matching detections to truth for the PR curve...", flush=True)
    for frame_index in range(FRAMES):
        p0, p1 = int(bounds[frame_index]), int(bounds[frame_index + 1])
        if p0 == p1:
            continue
        px = pred_x[p0:p1]
        py = pred_y[p0:p1]
        conf = confidence[p0:p1]
        dx = px[:, None] - truth["x"][frame_index][None, :]
        dy = py[:, None] - truth["y"][frame_index][None, :]
        distance2 = dx * dx + dy * dy
        matched_truth = np.zeros(TRUTH_PER_FRAME, dtype=bool)
        for local_pred in np.argsort(-conf, kind="stable"):
            candidates = np.flatnonzero(
                (distance2[local_pred] <= XY_GATE_PX**2) & ~matched_truth
            )
            if not len(candidates):
                continue
            local_truth = candidates[np.argmin(distance2[local_pred, candidates])]
            matched_truth[local_truth] = True
            flags[p0 + local_pred] = True
        if (frame_index + 1) % 1000 == 0:
            print(f"  PR matching: {frame_index + 1}/{FRAMES} frames", flush=True)
    return detection, flags

def precision_recall(
    confidence: np.ndarray,
    true_positive: np.ndarray,
    truth_count: int,
) -> dict[str, np.ndarray | float | int]:
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_tp = true_positive[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_tp)
    cumulative_fp = np.cumsum(1 - sorted_tp)
    recall = cumulative_tp / float(truth_count)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[1.0, precision, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    return {
        "confidence": sorted_confidence,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "average_precision": ap,
        "best_index": best,
        "best_confidence": float(sorted_confidence[best]),
        "best_f1": float(f1[best]),
    }

def match_tracks(
    truth: dict[str, np.ndarray], tracks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    counts = np.bincount(tracks["frame"], minlength=FRAMES)
    bounds = np.r_[0, np.cumsum(counts)]
    capacity = FRAMES * TRUTH_PER_FRAME
    output = {
        "xy_um": np.empty(capacity, dtype=np.float64),
        "z_um": np.empty(capacity, dtype=np.float64),
        "diameter_relative": np.empty(capacity, dtype=np.float64),
        "post_out_prob": np.empty(capacity, dtype=np.float64),
    }
    cursor = 0
    print("Matching trajectories by unsmoothed XY; measuring smoothed values...", flush=True)
    for frame_index in range(FRAMES):
        p0, p1 = int(bounds[frame_index]), int(bounds[frame_index + 1])
        if p0 == p1:
            continue
        raw_x = tracks["raw_x"][p0:p1]
        raw_y = tracks["raw_y"][p0:p1]
        dx = truth["x"][frame_index][:, None] - raw_x[None, :]
        dy = truth["y"][frame_index][:, None] - raw_y[None, :]
        distance2 = dx * dx + dy * dy
        cost = np.where(distance2 <= XY_GATE_PX**2, distance2, 1.0e9)
        truth_index, pred_index = linear_sum_assignment(cost)
        valid = distance2[truth_index, pred_index] <= XY_GATE_PX**2
        truth_index = truth_index[valid]
        pred_index = pred_index[valid]
        count = len(truth_index)
        if not count:
            continue
        absolute_pred = p0 + pred_index
        sx = tracks["smooth_x"][absolute_pred]
        sy = tracks["smooth_y"][absolute_pred]
        sz = tracks["smooth_z"][absolute_pred]
        diameter_um = tracks["diameter_px"][absolute_pred] * PIXEL_UM
        true_x = truth["x"][frame_index][truth_index]
        true_y = truth["y"][frame_index][truth_index]
        true_z = truth["z"][frame_index][truth_index]
        true_diameter = truth["diameter"][frame_index][truth_index]
        target = slice(cursor, cursor + count)
        output["xy_um"][target] = np.hypot(sx - true_x, sy - true_y) * PIXEL_UM
        output["z_um"][target] = (sz - true_z) * SLICE_UM
        output["diameter_relative"][target] = (diameter_um - true_diameter) / true_diameter
        output["post_out_prob"][target] = tracks["post_out_prob"][absolute_pred]
        cursor += count
        if (frame_index + 1) % 1000 == 0:
            print(f"  trajectory matching: {frame_index + 1}/{FRAMES} frames", flush=True)
    for key in output:
        output[key] = output[key][:cursor]
    output["matched_count"] = np.asarray([cursor], dtype=np.int64)
    output["predicted_count"] = np.asarray([len(tracks["frame"])], dtype=np.int64)
    return output
