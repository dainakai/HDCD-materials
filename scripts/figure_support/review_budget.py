"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

import math

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
SCENE_COUNT = 14
TOP_STRATUM = "top_score"
TARGET_COVERAGE = 0.95
OPERATING_POINTS = (20, 30, 40, 60, 63)
ORANGE = "#e66101"
BLUE = "#2b83ba"
GRAY = "#6f6f6f"

def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 13.0,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
        }
    )

def validate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    required = {
        "top_n_per_scene",
        "candidates_reviewed",
        "cumulative_confirmed_collisions",
        "cumulative_non_collisions",
        "cumulative_ambiguous",
        "top_n_precision",
    }
    missing = sorted(required.difference(curve.columns))
    if missing:
        raise RuntimeError(f"top-N curve is missing required columns: {missing}")

    frame = curve.copy()
    integer_columns = (
        "top_n_per_scene",
        "candidates_reviewed",
        "cumulative_confirmed_collisions",
        "cumulative_non_collisions",
        "cumulative_ambiguous",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.allclose(numeric, np.rint(numeric)):
            raise RuntimeError(f"{column} must contain integers")
        frame[column] = np.rint(numeric).astype(int)
    frame["top_n_precision"] = pd.to_numeric(
        frame["top_n_precision"], errors="raise"
    ).astype(float)
    frame = frame.sort_values("top_n_per_scene", kind="mergesort").reset_index(
        drop=True
    )

    expected_ranks = np.arange(1, int(frame["top_n_per_scene"].max()) + 1)
    if not np.array_equal(frame["top_n_per_scene"].to_numpy(int), expected_ranks):
        raise RuntimeError("top-N curve must contain complete ranks beginning at one")
    expected_reviews = frame["top_n_per_scene"].to_numpy(int) * SCENE_COUNT
    if not np.array_equal(
        frame["candidates_reviewed"].to_numpy(int), expected_reviews
    ):
        raise RuntimeError("review count must equal scene count times per-scene rank")
    status_total = (
        frame["cumulative_confirmed_collisions"]
        + frame["cumulative_non_collisions"]
        + frame["cumulative_ambiguous"]
    )
    if not np.array_equal(
        status_total.to_numpy(int), frame["candidates_reviewed"].to_numpy(int)
    ):
        raise RuntimeError("cumulative status counts do not equal review count")
    expected_precision = (
        frame["cumulative_confirmed_collisions"] / frame["candidates_reviewed"]
    )
    if not np.allclose(expected_precision, frame["top_n_precision"]):
        raise RuntimeError("stored top-N precision is inconsistent with counts")
    if not frame["cumulative_confirmed_collisions"].is_monotonic_increasing:
        raise RuntimeError("cumulative confirmed-collision count must be monotonic")
    if max(OPERATING_POINTS) > int(frame["top_n_per_scene"].max()):
        raise RuntimeError("an operating point lies outside the reviewed rank range")
    return frame

def validate_labels(labels: pd.DataFrame, total_confirmed: int) -> pd.DataFrame:
    required = {"scene_num", "review_score_rank", "review_stratum", "status"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise RuntimeError(f"review labels are missing required columns: {missing}")
    top = labels.loc[labels["review_stratum"].eq(TOP_STRATUM)].copy()
    top["scene_num"] = pd.to_numeric(top["scene_num"], errors="raise").astype(int)
    top["review_score_rank"] = pd.to_numeric(
        top["review_score_rank"], errors="raise"
    ).astype(int)
    if int(top["status"].eq("collision").sum()) != total_confirmed:
        raise RuntimeError("confirmed-collision total differs between labels and curve")
    if top["scene_num"].nunique() != SCENE_COUNT:
        raise RuntimeError("top-score labels must contain fourteen scenes")
    return top

def build_operating_table(
    curve: pd.DataFrame,
    top_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    total_confirmed = int(curve.iloc[-1]["cumulative_confirmed_collisions"])
    curve = curve.copy()
    curve["confirmed_tp_coverage"] = (
        curve["cumulative_confirmed_collisions"] / total_confirmed
    )
    minimum_target_row = curve.loc[
        curve["confirmed_tp_coverage"].ge(TARGET_COVERAGE)
    ].iloc[0]
    minimum_target_rank = int(minimum_target_row["top_n_per_scene"])
    suggested_rank = int(math.ceil(minimum_target_rank / 10.0) * 10)
    if suggested_rank not in OPERATING_POINTS:
        raise RuntimeError(
            f"rounded suggested rank {suggested_rank} is absent from operating points"
        )

    rows = []
    for rank in OPERATING_POINTS:
        point = curve.loc[curve["top_n_per_scene"].eq(rank)].iloc[0]
        later_confirmed = top_labels.loc[
            top_labels["status"].eq("collision")
            & top_labels["review_score_rank"].gt(rank)
        ]
        rows.append(
            {
                "reviews_per_scene": int(rank),
                "total_reviews": int(point["candidates_reviewed"]),
                "confirmed_tp": int(point["cumulative_confirmed_collisions"]),
                "confirmed_tp_coverage": float(point["confirmed_tp_coverage"]),
                "top_n_precision": float(point["top_n_precision"]),
                "later_confirmed_tp": int(len(later_confirmed)),
                "scenes_with_later_confirmed_tp": int(
                    later_confirmed["scene_num"].nunique()
                ),
                "suggested": bool(rank == suggested_rank),
            }
        )
    return pd.DataFrame(rows), minimum_target_rank, suggested_rank

def style_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(False)
    axis.tick_params(direction="out", width=0.8)

def plot_figure(
    curve: pd.DataFrame,
    operating: pd.DataFrame,
    suggested_rank: int,
    output_stem: Path,
) -> None:
    configure_plotting()
    total_confirmed = int(curve.iloc[-1]["cumulative_confirmed_collisions"])
    x = curve["top_n_per_scene"].to_numpy(int)
    coverage = (
        curve["cumulative_confirmed_collisions"].to_numpy(float)
        / total_confirmed
        * 100.0
    )
    precision = curve["top_n_precision"].to_numpy(float) * 100.0

    figure = plt.figure(figsize=(9.2, 5.4))
    axis = figure.add_subplot(1, 1, 1)
    figure.subplots_adjust(
        left=0.105,
        right=0.975,
        top=0.80,
        bottom=0.15,
    )

    axis.step(
        np.concatenate(([0], x)),
        np.concatenate(([0.0], coverage)),
        where="post",
        color=ORANGE,
        linewidth=2.2,
        label=f"Coverage of {total_confirmed} confirmed collisions",
    )
    axis.plot(
        x,
        precision,
        color=BLUE,
        linewidth=2.2,
        label="Top-N precision",
    )

    axis.axvline(
        suggested_rank,
        color=GRAY,
        linewidth=1.5,
        linestyle="-",
        zorder=0,
    )

    suggested = operating.loc[
        operating["reviews_per_scene"].eq(suggested_rank)
    ].iloc[0]
    axis.annotate(
        f"{100.0 * suggested['confirmed_tp_coverage']:.1f}%",
        xy=(suggested_rank, suggested["confirmed_tp_coverage"] * 100.0),
        xytext=(8.0, -11.0),
        textcoords="offset points",
        ha="left",
        va="top",
        color=ORANGE,
        fontsize=11.0,
    )
    axis.annotate(
        f"{100.0 * suggested['top_n_precision']:.1f}%",
        xy=(suggested_rank, suggested["top_n_precision"] * 100.0),
        xytext=(8.0, 8.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=BLUE,
        fontsize=11.0,
    )

    axis.set_xlim(0.0, 80.0)
    axis.set_ylim(0.0, 104.0)
    axis.set_xlabel("Candidates reviewed per scene")
    axis.set_ylabel("Coverage and precision [%]")
    axis.set_xticks(np.arange(0, 81, 10))
    axis.yaxis.set_major_locator(mticker.MultipleLocator(20.0))
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
    style_axis(axis)

    secondary = axis.secondary_xaxis(
        "top",
        functions=(
            lambda per_scene: per_scene * SCENE_COUNT,
            lambda total: total / SCENE_COUNT,
        ),
    )
    secondary.set_xlabel("Total candidates reviewed across 14 scenes")
    secondary.set_xticks(np.arange(0, 81, 20) * SCENE_COUNT)
    secondary.tick_params(direction="out", width=0.8)

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        ncols=2,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
