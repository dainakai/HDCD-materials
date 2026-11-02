#!/usr/bin/env python3
"""Plot a paper-ready case study showing why image features are useful.

The figure compares two held-out synthetic candidates with similarly high
77-feature trajectory scores:

* a ground-truth collision, shown before, at, and after coalescence; and
* a rejected tracking artifact in which two reconstructed track IDs refer to
  one source particle.

The reconstructed intensity panels use the same phase-retrieval and depth-
propagation implementation as the frozen image-feature pipeline.  The score
panel compares the trajectory-only score with the 92-feature base score.
This script writes only figure/provenance artifacts; it does not edit the
manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASET_NAME = (
    "collision_benchmark_260715_200p_6000f_clean_2000coll_"
    "halfmix25_v1_outside256_yjitter10_seed260745"
)
SCENE = "C000H001S0011"
SCENE_KEY = f"single:{SCENE}"
FPS = 4000.0
PIXEL_SIZE_UM = 10.0
ARIAL_COMPATIBLE_FONT = Path(
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
)

TERMINATING_COLOR = "#D84A3A"
CONTINUING_COLOR = "#2C6FB7"
TRAJECTORY_COLOR = "#6B747C"
IMAGE_SCORE_COLOR = "#009E73"
INK = "#263238"
MUTED_INK = "#536168"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    vanished_track_id: int
    survivor_track_id: int
    candidate_frame: int
    display_frames: tuple[int, int, int]
    time_reference_frame: int
    truth_event_id: int | None = None
    expected_shared_source_id: int | None = None

    @property
    def candidate_key(self) -> str:
        return (
            f"{self.vanished_track_id}_{self.survivor_track_id}_"
            f"{self.candidate_frame}"
        )


TRUE_CASE = CaseSpec(
    key="true_collision",
    title="Ground-truth collision (event 1165): 212 + 250 → 293 µm",
    vanished_track_id=12515,
    survivor_track_id=12410,
    candidate_frame=3859,
    display_frames=(3855, 3859, 3863),
    time_reference_frame=3859,
    truth_event_id=1165,
)

ARTIFACT_CASE = CaseSpec(
    key="shared_particle_artifact",
    title="Shared-particle artifact",
    vanished_track_id=17742,
    survivor_track_id=17721,
    candidate_frame=5331,
    display_frames=(5327, 5331, 5335),
    time_reference_frame=5331,
    expected_shared_source_id=7332,
)

CASES = (TRUE_CASE, ARTIFACT_CASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/media/dai-server/DATAM2/repos/260727"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/media/dai-server/M2Disk2TB/repos/260712"),
        help="Repository containing the frozen holographic reconstruction code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "figures"
            / "image_feature_case_study"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device. Use CUDA_VISIBLE_DEVICES to select a physical GPU.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    if not ARIAL_COMPATIBLE_FONT.exists():
        raise FileNotFoundError(
            f"Arial-compatible font not found: {ARIAL_COMPATIBLE_FONT}"
        )
    font_manager.fontManager.addfont(str(ARIAL_COMPATIBLE_FONT))
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
            "pdf.use14corefonts": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def unique_candidate_row(df: pd.DataFrame, case: CaseSpec) -> pd.Series:
    rows = df[
        df["vanished_track_id"].eq(case.vanished_track_id)
        & df["survivor_track_id"].eq(case.survivor_track_id)
        & df["candidate_frame"].eq(case.candidate_frame)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one row for {case.candidate_key}, found {len(rows)}"
        )
    return rows.iloc[0]


def policy_contains_candidate(policy: pd.DataFrame, case: CaseSpec) -> bool:
    mask = (
        policy["vanished_track_id"].eq(case.vanished_track_id)
        & policy["survivor_track_id"].eq(case.survivor_track_id)
        & policy["candidate_frame"].eq(case.candidate_frame)
    )
    return bool(mask.any())


def truth_matches_candidate(
    truth: pd.DataFrame,
    row: pd.Series,
    *,
    frame_tolerance: int = 5,
    xy_tolerance_um: float = 350.0,
    z_tolerance_um: float = 15000.0,
) -> pd.DataFrame:
    local = truth[
        truth["frame"].between(
            int(row["candidate_frame"]) - frame_tolerance,
            int(row["candidate_frame"]) + frame_tolerance,
        )
    ].copy()
    dxy = np.hypot(
        local["x_um"].to_numpy(float) - float(row["x_um"]),
        local["y_um"].to_numpy(float) - float(row["y_um"]),
    )
    dz = np.abs(local["z_um"].to_numpy(float) - float(row["z_um"]))
    return local[(dxy <= xy_tolerance_um) & (dz <= z_tolerance_um)]


def load_and_audit_cases(
    experiment_root: Path,
) -> tuple[
    Path,
    Path,
    dict[str, pd.Series],
    dict[str, pd.Series],
    pd.DataFrame,
    dict[str, Any],
]:
    dataset_dir = experiment_root / "synth" / DATASET_NAME
    evaluation_dir = (
        experiment_root
        / "data"
        / "retraincurrent_pr90_v2"
        / "final_evaluation"
        / "seed260745"
    )
    track_dir = (
        experiment_root
        / "runs"
        / "retrainv20"
        / "reconstruction"
        / "seed260745"
        / "best_tiebreak_z010"
        / "smoothedtrajs"
    )

    required_paths = (
        dataset_dir,
        track_dir,
        evaluation_dir / "candidate_scores_before_truth.csv.gz",
        evaluation_dir / "image_enriched.csv.gz",
        evaluation_dir / "image_same_frame.csv",
        evaluation_dir / "policy_predictions_before_truth.csv.gz",
        evaluation_dir / "image_feature_summary.json",
        dataset_dir / "labels" / "collision_events.csv",
        experiment_root
        / "data"
        / "relation_attention_3d_vs_2d"
        / "mappings"
        / "seed260745"
        / "track_truth_mapping.csv.gz",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    scores = pd.read_csv(
        evaluation_dir / "candidate_scores_before_truth.csv.gz",
        usecols=[
            "vanished_track_id",
            "survivor_track_id",
            "candidate_frame",
            "x_um",
            "y_um",
            "z_um",
            "static_score",
            "full_score",
            "image_enriched",
        ],
    )
    image_features = pd.read_csv(
        evaluation_dir / "image_enriched.csv.gz",
        usecols=[
            "candidate_key",
            "vanished_track_id",
            "survivor_track_id",
            "candidate_frame",
            "vanished_end_frame",
            "rendered_pre_frames",
            "pre_same_interp_count",
            "pre_same_observed_count",
            "pre_alternating_same_count",
        ],
    )
    candidate_rows = {case.key: unique_candidate_row(scores, case) for case in CASES}
    feature_rows = {
        case.key: unique_candidate_row(image_features, case) for case in CASES
    }

    for case in CASES:
        if int(candidate_rows[case.key]["image_enriched"]) != 1:
            raise RuntimeError(f"{case.candidate_key} lacks image features")
        if int(feature_rows[case.key]["rendered_pre_frames"]) != 9:
            raise RuntimeError(f"{case.candidate_key} does not have nine rendered frames")

    trajectory_delta = abs(
        float(candidate_rows[TRUE_CASE.key]["static_score"])
        - float(candidate_rows[ARTIFACT_CASE.key]["static_score"])
    )
    if trajectory_delta >= 0.10:
        raise RuntimeError(
            f"selected trajectory scores are no longer similarly high: delta={trajectory_delta}"
        )
    if int(feature_rows[TRUE_CASE.key]["pre_same_interp_count"]) != 0:
        raise RuntimeError("true case image-component count changed")
    if int(feature_rows[ARTIFACT_CASE.key]["pre_same_interp_count"]) != 9:
        raise RuntimeError("artifact image-component count changed")

    truth = pd.read_csv(dataset_dir / "labels" / "collision_events.csv")
    true_truth = truth[truth["collision_event_id"].eq(TRUE_CASE.truth_event_id)]
    if len(true_truth) != 1 or int(true_truth.iloc[0]["frame"]) != 3859:
        raise RuntimeError("ground-truth collision event 1165 is unavailable or changed")
    if TRUE_CASE.truth_event_id not in set(
        truth_matches_candidate(truth, candidate_rows[TRUE_CASE.key])[
            "collision_event_id"
        ].astype(int)
    ):
        raise RuntimeError("true candidate no longer matches event 1165")
    if not truth_matches_candidate(truth, candidate_rows[ARTIFACT_CASE.key]).empty:
        raise RuntimeError("artifact candidate unexpectedly matches a truth collision")

    policy = pd.read_csv(
        evaluation_dir / "policy_predictions_before_truth.csv.gz",
        usecols=[
            "vanished_track_id",
            "survivor_track_id",
            "candidate_frame",
            "x_um",
            "y_um",
            "z_um",
        ],
    )
    true_candidate_retained = policy_contains_candidate(policy, TRUE_CASE)
    if not true_candidate_retained:
        raise RuntimeError("true candidate is unexpectedly absent from final predictions")
    if policy_contains_candidate(policy, ARTIFACT_CASE):
        raise RuntimeError("artifact candidate is unexpectedly present in final predictions")

    truth_event = true_truth.iloc[0]
    policy_near_truth = policy[
        policy["candidate_frame"].between(
            int(truth_event["frame"]) - 5,
            int(truth_event["frame"]) + 5,
        )
    ].copy()
    policy_dxy = np.hypot(
        policy_near_truth["x_um"].to_numpy(float) - float(truth_event["x_um"]),
        policy_near_truth["y_um"].to_numpy(float) - float(truth_event["y_um"]),
    )
    policy_dz = np.abs(
        policy_near_truth["z_um"].to_numpy(float) - float(truth_event["z_um"])
    )
    true_event_detected = bool(((policy_dxy <= 350.0) & (policy_dz <= 15000.0)).any())
    if not true_event_detected:
        raise RuntimeError("ground-truth event 1165 is no longer represented downstream")

    mapping_path = (
        experiment_root
        / "data"
        / "relation_attention_3d_vs_2d"
        / "mappings"
        / "seed260745"
        / "track_truth_mapping.csv.gz"
    )
    mapping = pd.read_csv(
        mapping_path,
        usecols=["frame", "observed_track_id", "source_particle_id"],
    )
    artifact_sources: list[set[int]] = []
    for track_id in (
        ARTIFACT_CASE.vanished_track_id,
        ARTIFACT_CASE.survivor_track_id,
    ):
        values = mapping.loc[
            mapping["observed_track_id"].eq(track_id)
            & mapping["frame"].between(
                ARTIFACT_CASE.candidate_frame - 10,
                ARTIFACT_CASE.candidate_frame,
            )
            & mapping["source_particle_id"].ge(0),
            "source_particle_id",
        ]
        artifact_sources.append(set(values.astype(int)))
    shared_sources = artifact_sources[0] & artifact_sources[1]
    if shared_sources != {ARTIFACT_CASE.expected_shared_source_id}:
        raise RuntimeError(f"unexpected shared source particles: {shared_sources}")

    parent_ids = {
        int(truth_event["parent1_id"]),
        int(truth_event["parent2_id"]),
    }
    child_id = int(truth_event["child_id"])
    true_mapping = mapping[
        mapping["observed_track_id"].isin(
            [TRUE_CASE.vanished_track_id, TRUE_CASE.survivor_track_id]
        )
        & mapping["frame"].between(
            int(truth_event["frame"]) - 15,
            int(truth_event["frame"]) + 15,
        )
    ]
    vanished_sources = set(
        true_mapping.loc[
            true_mapping["observed_track_id"].eq(TRUE_CASE.vanished_track_id),
            "source_particle_id",
        ].astype(int)
    )
    survivor_sources = set(
        true_mapping.loc[
            true_mapping["observed_track_id"].eq(TRUE_CASE.survivor_track_id),
            "source_particle_id",
        ].astype(int)
    )
    if not vanished_sources.issubset(parent_ids):
        raise RuntimeError("terminating true track is not mapped to a collision parent")
    if not (survivor_sources & parent_ids) or child_id not in survivor_sources:
        raise RuntimeError("continuing true track does not connect parent to child")

    with (evaluation_dir / "image_feature_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        feature_summary = json.load(handle)
    same_config = feature_summary["same_component"]
    expected_config = {
        "pre_frames": 10,
        "evaluated_relative_frames": [-10, -2],
        "threshold": 30,
        "view_size": 128,
        "priter": 12,
        "image_frame_offset": 1,
    }
    for key, expected in expected_config.items():
        if same_config[key] != expected:
            raise RuntimeError(
                f"image-feature configuration changed: {key}={same_config[key]}"
            )

    frame_table = pd.read_csv(
        evaluation_dir / "image_same_frame.csv",
        usecols=[
            "candidate_key",
            "frame",
            "rendered",
            "interp_same_component",
            "center_x",
            "center_y",
            "center_z",
        ],
        low_memory=False,
    )
    frame_table = frame_table[frame_table["candidate_key"].isin(
        [case.candidate_key for case in CASES]
    )].copy()

    audit = {
        "trajectory_score_difference": trajectory_delta,
        "artifact_shared_source_particle_id": int(
            ARTIFACT_CASE.expected_shared_source_id
        ),
        "artifact_absent_from_final_predictions": True,
        "true_candidate_retained_in_final_predictions": true_candidate_retained,
        "true_event_has_downstream_prediction": true_event_detected,
        "truth_parent_ids": sorted(parent_ids),
        "truth_child_id": child_id,
        "truth_parent_diameters_um": [
            float(truth_event["parent1_diameter_um"]),
            float(truth_event["parent2_diameter_um"]),
        ],
        "truth_child_diameter_um": float(truth_event["child_diameter_um"]),
        "score_source": str(
            evaluation_dir / "candidate_scores_before_truth.csv.gz"
        ),
        "image_feature_source": str(evaluation_dir / "image_enriched.csv.gz"),
        "image_frame_source": str(evaluation_dir / "image_same_frame.csv"),
        "truth_source": str(dataset_dir / "labels" / "collision_events.csv"),
        "mapping_source": str(mapping_path),
        "policy_source": str(
            evaluation_dir / "policy_predictions_before_truth.csv.gz"
        ),
        "image_feature_config": expected_config,
    }
    return (
        dataset_dir,
        track_dir,
        candidate_rows,
        feature_rows,
        frame_table,
        audit,
    )


def load_reference_api(reference_root: Path) -> dict[str, Any]:
    if not (reference_root / "src").is_dir():
        raise FileNotFoundError(f"reference source tree not found: {reference_root}")
    sys.path.insert(0, str(reference_root))
    import torch
    from src.collision import build_enhanced_candidate_features as enhanced
    from src.collision.generate_collision_gifs import (
        EventSpec,
        choose_center,
        determine_mode,
        load_track,
        point_at,
    )

    return {
        "torch": torch,
        "enhanced": enhanced,
        "EventSpec": EventSpec,
        "choose_center": choose_center,
        "determine_mode": determine_mode,
        "load_track": load_track,
        "point_at": point_at,
    }


def crop_with_black_padding(
    image: np.ndarray,
    cx: float,
    cy: float,
    size: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Center the display crop and fill the unrecorded region with black."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2D reconstruction, got shape {image.shape}")
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.shape[1], x1)
    src_y1 = min(image.shape[0], y1)
    crop = np.zeros((size, size), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        height = src_y1 - src_y0
        width = src_x1 - src_x0
        crop[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = image[
            src_y0:src_y1,
            src_x0:src_x1,
        ]
        valid[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = True
    return crop, float(x0), float(y0), valid


def normalize_valid_crop(crop: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize recorded pixels only, leaving padded pixels exactly black."""
    values = crop[valid]
    if values.size == 0:
        return np.zeros_like(crop, dtype=np.uint8)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    normalized = np.zeros_like(crop, dtype=np.uint8)
    if hi > lo:
        scaled = np.rint(
            np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        normalized[valid] = scaled
    return normalized


def reconstruct_case_panels(
    case: CaseSpec,
    *,
    api: dict[str, Any],
    context: Any,
    reconstruction_args: SimpleNamespace,
    device: Any,
    track_dir: Path,
    frame_table: pd.DataFrame,
    view_size: int,
) -> tuple[list[dict[str, Any]], Any, Any]:
    load_track = api["load_track"]
    point_at = api["point_at"]
    EventSpec = api["EventSpec"]
    vanished = load_track(track_dir, case.vanished_track_id)
    survivor = load_track(track_dir, case.survivor_track_id)
    mode = api["determine_mode"](
        track_dir,
        case.candidate_frame,
        case.survivor_track_id,
        10,
    )
    event = EventSpec(
        scene=SCENE,
        event_id=case.truth_event_id or 0,
        label=case.key,
        frame=case.candidate_frame,
        vanished_track_id=case.vanished_track_id,
        survivor_track_id=case.survivor_track_id,
        vanished_diameter_um=None,
        survivor_diameter_um=None,
        collision_score=None,
        mode=mode,
    )

    panels: list[dict[str, Any]] = []
    enhanced = api["enhanced"]
    for frame in case.display_frames:
        stored = frame_table[
            frame_table["candidate_key"].eq(case.candidate_key)
            & frame_table["frame"].eq(frame)
        ]
        if len(stored) == 1:
            center_x = float(stored.iloc[0]["center_x"])
            center_y = float(stored.iloc[0]["center_y"])
            center_z = float(stored.iloc[0]["center_z"])
            center_source = "frozen_image_feature_table"
        elif len(stored) == 0:
            center = api["choose_center"](event, vanished, survivor, frame)
            center_x, center_y, center_z = center.x, center.y, center.z
            center_source = "continuing_track"
        else:
            raise RuntimeError(
                f"duplicate stored frame centers for {case.candidate_key}, frame {frame}"
            )

        with api["torch"].no_grad():
            frame_fft = enhanced.phase_fft_for_frame(
                context,
                frame,
                reconstruction_args,
                device,
            )
            full = enhanced.reconstruct_intensity_from_fft(
                context,
                frame_fft,
                center_z,
                device,
            )
        crop, x0, y0, valid = crop_with_black_padding(
            full,
            center_x,
            center_y,
            view_size,
        )
        normalized = normalize_valid_crop(crop, valid)
        panels.append(
            {
                "frame": frame,
                "elapsed_ms": (frame - case.time_reference_frame) / FPS * 1000.0,
                "image": normalized,
                "x0": x0,
                "y0": y0,
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "center_source": center_source,
                "recorded_fraction": float(np.mean(valid)),
                "vanished_point": point_at(vanished, frame, nearest=False),
                "survivor_point": point_at(survivor, frame, nearest=False),
            }
        )
        del frame_fft, full, crop, valid, normalized
    return panels, vanished, survivor


def overlay_track_tail(
    ax: plt.Axes,
    track: pd.DataFrame,
    frame: int,
    x0: float,
    y0: float,
    color: str,
) -> None:
    tail = track[track["frame"].between(frame - 4, frame)].sort_values("frame")
    if tail.empty:
        return
    ax.plot(
        tail["x"].to_numpy(float) - x0,
        tail["y"].to_numpy(float) - y0,
        color=color,
        linewidth=1.3,
        alpha=1.0,
        solid_capstyle="round",
        zorder=5,
    )


def overlay_track_marker(
    ax: plt.Axes,
    point: Any | None,
    x0: float,
    y0: float,
    *,
    color: str,
    marker: str,
) -> None:
    if point is None:
        return
    x = float(point.x) - x0
    y = float(point.y) - y0
    ax.scatter(
        [x],
        [y],
        s=27,
        marker=marker,
        facecolors=color,
        edgecolors="none",
        linewidths=0.0,
        zorder=7,
    )


def add_scale_bar(ax: plt.Axes, view_size: int) -> None:
    scale_bar_px = 500.0 / PIXEL_SIZE_UM
    x0 = 8.0
    y = float(view_size - 10)
    bar = ax.plot(
        [x0, x0 + scale_bar_px],
        [y, y],
        color="white",
        linewidth=2.7,
        solid_capstyle="butt",
        zorder=8,
    )[0]
    bar.set_path_effects(
        [path_effects.Stroke(linewidth=4.2, foreground=INK), path_effects.Normal()]
    )
    label = ax.text(
        x0 + scale_bar_px / 2.0,
        y - 4.5,
        "500 µm",
        ha="center",
        va="bottom",
        color="white",
        fontsize=6.3,
        zorder=9,
    )
    label.set_path_effects(
        [path_effects.Stroke(linewidth=2.1, foreground=INK), path_effects.Normal()]
    )


def time_title(case: CaseSpec, elapsed_ms: float) -> str:
    elapsed = f"{elapsed_ms:+.2f}".replace("-", "−")
    if elapsed_ms < 0.0:
        return f"pre  {elapsed} ms"
    if elapsed_ms > 0.0:
        return f"post  {elapsed} ms"
    reference = "collision" if case.key == TRUE_CASE.key else "candidate"
    return f"{reference}  0.00 ms"


def add_image_panel(
    ax: plt.Axes,
    panel: dict[str, Any],
    *,
    case: CaseSpec,
    vanished: pd.DataFrame,
    survivor: pd.DataFrame,
    view_size: int,
    show_scale_bar: bool,
) -> None:
    ax.imshow(
        panel["image"],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    if panel["vanished_point"] is not None:
        overlay_track_tail(
            ax,
            vanished,
            int(panel["frame"]),
            float(panel["x0"]),
            float(panel["y0"]),
            TERMINATING_COLOR,
        )
    if panel["survivor_point"] is not None:
        overlay_track_tail(
            ax,
            survivor,
            int(panel["frame"]),
            float(panel["x0"]),
            float(panel["y0"]),
            CONTINUING_COLOR,
        )
    overlay_track_marker(
        ax,
        panel["vanished_point"],
        float(panel["x0"]),
        float(panel["y0"]),
        color=TERMINATING_COLOR,
        marker="o",
    )
    overlay_track_marker(
        ax,
        panel["survivor_point"],
        float(panel["x0"]),
        float(panel["y0"]),
        color=CONTINUING_COLOR,
        marker="s",
    )
    if show_scale_bar:
        add_scale_bar(ax, view_size)
    ax.set_xlim(-0.5, view_size - 0.5)
    ax.set_ylim(view_size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(INK)
    ax.set_title(
        time_title(case, float(panel["elapsed_ms"])),
        pad=2.5,
        color=INK,
        fontsize=8.0,
        fontweight="normal",
    )


def add_score_panel(
    ax: plt.Axes,
    candidate_rows: dict[str, pd.Series],
) -> None:
    trajectory = np.array(
        [
            float(candidate_rows[TRUE_CASE.key]["static_score"]),
            float(candidate_rows[ARTIFACT_CASE.key]["static_score"]),
        ]
    )
    image_score = np.array(
        [
            float(candidate_rows[TRUE_CASE.key]["full_score"]),
            float(candidate_rows[ARTIFACT_CASE.key]["full_score"]),
        ]
    )
    group_y = np.array([1.0, 0.0])
    offset = 0.105
    height = 0.14
    trajectory_y = group_y + offset
    image_y = group_y - offset

    trajectory_bars = ax.barh(
        trajectory_y,
        trajectory,
        height=height,
        color=TRAJECTORY_COLOR,
        edgecolor="none",
        label="trajectory score",
        zorder=3,
    )
    image_bars = ax.barh(
        image_y,
        image_score,
        height=height,
        color=IMAGE_SCORE_COLOR,
        edgecolor="none",
        label="+ image features",
        zorder=3,
    )

    for bars, values, label_color in (
        (trajectory_bars, trajectory, TRAJECTORY_COLOR),
        (image_bars, image_score, IMAGE_SCORE_COLOR),
    ):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                float(value) + 0.014,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.3f}",
                ha="left",
                va="center",
                color=label_color,
                fontsize=7.4,
                zorder=4,
            )

    for y, label in (
        (1.0, "True collision  (retained)"),
        (0.0, "Shared-particle artifact  (rejected)"),
    ):
        ax.text(
            0.045,
            y + 0.30,
            label,
            ha="left",
            va="center",
            color=INK,
            fontsize=7.8,
        )

    ax.set_xlim(0.0, 1.10)
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Candidate score")
    ax.grid(False)
    ax.axvline(
        0.0,
        color="#111111",
        linewidth=0.85,
        zorder=2,
        clip_on=False,
    )
    ax.tick_params(axis="x", colors=MUTED_INK, length=2.5, width=0.55)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.45,
        borderaxespad=0.0,
        columnspacing=0.85,
        fontsize=7.2,
    )


def build_figure(
    reconstructed: dict[str, tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]],
    candidate_rows: dict[str, pd.Series],
    *,
    view_size: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 4.30), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.62),
        left=0.025,
        right=0.985,
        bottom=0.205,
        top=0.885,
        wspace=0.10,
        hspace=0.78,
    )

    for row_index, case in enumerate(CASES):
        panels, vanished, survivor = reconstructed[case.key]
        axes = [fig.add_subplot(grid[row_index, column]) for column in range(3)]
        for column, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
            add_image_panel(
                ax,
                panel,
                case=case,
                vanished=vanished,
                survivor=survivor,
                view_size=view_size,
                show_scale_bar=(row_index == 0 and column == 0),
            )

        subcaption = (
            "(a) Ground-truth collision: 212 + 250 → 293 µm"
            if row_index == 0
            else "(b) Shared-particle artifact: two track IDs share one true particle"
        )
        axes[1].text(
            0.5,
            -0.23,
            subcaption,
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            color=INK,
            fontsize=8.0,
        )

    score_ax = fig.add_subplot(grid[:, 3])
    add_score_panel(score_ax, candidate_rows)
    score_ax.text(
        0.5,
        -0.145,
        "(c) Scores with and without image features",
        transform=score_ax.transAxes,
        ha="center",
        va="top",
        color=INK,
        fontsize=8.0,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TERMINATING_COLOR,
            linewidth=1.4,
            marker="o",
            markersize=4.6,
            markerfacecolor=TERMINATING_COLOR,
            markeredgecolor="none",
            markeredgewidth=0.0,
            label="terminating track",
        ),
        Line2D(
            [0],
            [0],
            color=CONTINUING_COLOR,
            linewidth=1.4,
            marker="s",
            markersize=4.5,
            markerfacecolor=CONTINUING_COLOR,
            markeredgecolor="none",
            markeredgewidth=0.0,
            label="continuing track",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.305, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.25,
    )
    return fig


def write_provenance(
    output_dir: Path,
    *,
    candidate_rows: dict[str, pd.Series],
    feature_rows: dict[str, pd.Series],
    reconstructed: dict[str, tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]],
    audit: dict[str, Any],
    device: str,
) -> None:
    manifest_rows: list[dict[str, Any]] = []
    for case in CASES:
        scores = candidate_rows[case.key]
        image = feature_rows[case.key]
        panels = reconstructed[case.key][0]
        manifest_rows.append(
            {
                "case": case.key,
                "candidate_key": case.candidate_key,
                "truth_collision_event_id": case.truth_event_id,
                "candidate_frame": case.candidate_frame,
                "time_reference_frame": case.time_reference_frame,
                "display_frames": ";".join(str(frame) for frame in case.display_frames),
                "vanished_track_id": case.vanished_track_id,
                "survivor_track_id": case.survivor_track_id,
                "shared_source_particle_id": case.expected_shared_source_id,
                "truth_parent1_diameter_um": (
                    audit["truth_parent_diameters_um"][0]
                    if case.key == TRUE_CASE.key
                    else None
                ),
                "truth_parent2_diameter_um": (
                    audit["truth_parent_diameters_um"][1]
                    if case.key == TRUE_CASE.key
                    else None
                ),
                "truth_child_diameter_um": (
                    audit["truth_child_diameter_um"]
                    if case.key == TRUE_CASE.key
                    else None
                ),
                "trajectory_score_p_traj": float(scores["static_score"]),
                "base_score_b_with_image": float(scores["full_score"]),
                "score_change": float(scores["full_score"])
                - float(scores["static_score"]),
                "pre_same_interp_count": int(image["pre_same_interp_count"]),
                "pre_same_observed_count": int(image["pre_same_observed_count"]),
                "pre_alternating_same_count": int(
                    image["pre_alternating_same_count"]
                ),
                "display_center_sources": ";".join(
                    str(panel["center_source"]) for panel in panels
                ),
                "recorded_fractions": ";".join(
                    f"{float(panel['recorded_fraction']):.6f}" for panel in panels
                ),
                "display_crop_policy": (
                    "128 px crop centered on the target; unrecorded pixels are black"
                ),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "case_study_manifest.csv",
        index=False,
    )

    provenance = {
        "description": (
            "Held-out synthetic case study comparing a true collision with a "
            "shared-particle tracking artifact. Hidden truth/mapping data are "
            "used only to audit and label the displayed cases."
        ),
        "device": device,
        "fps": FPS,
        "pixel_size_um": PIXEL_SIZE_UM,
        "normalization": (
            "per-panel 1st-to-99th percentile over recorded pixels only; "
            "unrecorded padding remains black"
        ),
        "display_crop_policy": (
            "128 px square centered on the stored feature center when available; "
            "regions outside the recorded image are displayed as black padding"
        ),
        "model_feature_note": (
            "The plotted image-feature counts and scores come from the frozen feature "
            "tables; display padding affects visualization only."
        ),
        "audit": audit,
        "cases": manifest_rows,
    }
    (output_dir / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_plotting()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        dataset_dir,
        track_dir,
        candidate_rows,
        feature_rows,
        frame_table,
        audit,
    ) = load_and_audit_cases(args.experiment_root)

    api = load_reference_api(args.reference_root)
    torch = api["torch"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")

    feature_config = audit["image_feature_config"]
    scene_summary_path = (
        args.experiment_root
        / "data"
        / "retraincurrent_pr90_v2"
        / "final_evaluation"
        / "seed260745"
        / "image_feature_summary.json"
    )
    with scene_summary_path.open("r", encoding="utf-8") as handle:
        scene_summary = json.load(handle)["scene_contexts"][SCENE_KEY]

    reconstruction_args = SimpleNamespace(
        data_dir=dataset_dir,
        track_dir=track_dir,
        data_scene=SCENE,
        bundle_num=int(scene_summary["bundle_num"]),
        glassplate_to_volume_um=float(scene_summary["glassplate_to_volume_um"]),
        padlen=2048,
        priter=int(feature_config["priter"]),
        image_frame_offset=int(feature_config["image_frame_offset"]),
    )
    context = api["enhanced"].get_scene_context(
        SCENE_KEY,
        pd.Series(dtype=object),
        reconstruction_args,
        {},
        device,
    )
    view_size = int(feature_config["view_size"])
    reconstructed: dict[
        str, tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]
    ] = {}
    for case in CASES:
        reconstructed[case.key] = reconstruct_case_panels(
            case,
            api=api,
            context=context,
            reconstruction_args=reconstruction_args,
            device=device,
            track_dir=track_dir,
            frame_table=frame_table,
            view_size=view_size,
        )

    figure = build_figure(
        reconstructed,
        candidate_rows,
        view_size=view_size,
    )
    pdf_path = args.output_dir / "image_feature_case_study_arial_compatible.pdf"
    png_path = args.output_dir / "image_feature_case_study_arial_compatible.png"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
        metadata={"Title": "Image-feature collision case study"},
    )
    figure.savefig(
        png_path,
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    plt.close(figure)

    write_provenance(
        args.output_dir,
        candidate_rows=candidate_rows,
        feature_rows=feature_rows,
        reconstructed=reconstructed,
        audit=audit,
        device=str(device),
    )
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
