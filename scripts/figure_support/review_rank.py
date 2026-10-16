"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from .scene_index import SCENE_NUMBERS
TOP_STRATUM='top_score'
STATUS_ORDER=('collision','non_collision','ambiguous')

def validate_top_score_cohort(labels: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "scene_num",
        "review_score_rank",
        "review_stratum",
        "review_inclusion_probability",
        "status",
        "collision_score",
    }
    missing = sorted(required_columns.difference(labels.columns))
    if missing:
        raise RuntimeError(f"review labels are missing required columns: {missing}")

    top = labels.loc[labels["review_stratum"].eq(TOP_STRATUM)].copy()
    top["scene_num"] = pd.to_numeric(top["scene_num"], errors="raise").astype(int)
    ranks = pd.to_numeric(top["review_score_rank"], errors="raise")
    if not np.allclose(ranks, np.rint(ranks)):
        raise RuntimeError("top-score review ranks must be integers")
    top["review_score_rank"] = np.rint(ranks).astype(int)
    top["collision_score"] = pd.to_numeric(
        top["collision_score"], errors="raise"
    ).astype(float)

    expected_scenes = set(int(scene) for scene in SCENE_NUMBERS)
    observed_scenes = set(int(scene) for scene in top["scene_num"].unique())
    if observed_scenes != expected_scenes:
        raise RuntimeError(
            "top-score cohort scene mismatch: "
            f"expected {sorted(expected_scenes)}, observed {sorted(observed_scenes)}"
        )

    duplicate_mask = top.duplicated(["scene_num", "review_score_rank"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = top.loc[
            duplicate_mask, ["scene_num", "review_score_rank"]
        ].to_dict("records")
        raise RuntimeError(f"duplicate scene/rank rows: {duplicate_rows[:10]}")

    rank_sets = {
        int(scene): tuple(sorted(group["review_score_rank"].astype(int).tolist()))
        for scene, group in top.groupby("scene_num", sort=True)
    }
    maximum_ranks = {ranks[-1] for ranks in rank_sets.values() if ranks}
    if len(maximum_ranks) != 1:
        raise RuntimeError(f"top-score rank maxima differ by scene: {maximum_ranks}")
    maximum_rank = maximum_ranks.pop()
    expected_ranks = tuple(range(1, maximum_rank + 1))
    invalid_rank_sets = {
        scene: ranks for scene, ranks in rank_sets.items() if ranks != expected_ranks
    }
    if invalid_rank_sets:
        raise RuntimeError(
            "top-score ranks must be complete and contiguous in every scene: "
            f"{invalid_rank_sets}"
        )

    probabilities = pd.to_numeric(
        top["review_inclusion_probability"], errors="raise"
    ).to_numpy(float)
    if not np.allclose(probabilities, 1.0):
        raise RuntimeError("top-score cohort must have inclusion probability one")

    statuses = set(str(value) for value in top["status"].dropna().unique())
    unexpected_statuses = sorted(statuses.difference(STATUS_ORDER))
    if unexpected_statuses or top["status"].isna().any():
        raise RuntimeError(
            "unexpected or missing review statuses: "
            f"unexpected={unexpected_statuses}, missing={int(top['status'].isna().sum())}"
        )

    for scene, group in top.groupby("scene_num", sort=True):
        ordered = group.sort_values("review_score_rank", kind="mergesort")
        score_differences = np.diff(ordered["collision_score"].to_numpy(float))
        if np.any(score_differences > 1.0e-12):
            raise RuntimeError(
                f"scene {scene}: collision scores increase with review score rank"
            )

    return top.sort_values(
        ["review_score_rank", "scene_num"], kind="mergesort"
    ).reset_index(drop=True)

def aggregate_by_rank(top: pd.DataFrame) -> pd.DataFrame:
    scene_count = len(SCENE_NUMBERS)
    maximum_rank = int(top["review_score_rank"].max())
    rows: list[dict[str, Any]] = []
    cumulative = {status: 0 for status in STATUS_ORDER}

    for rank in range(1, maximum_rank + 1):
        block = top.loc[top["review_score_rank"].eq(rank)]
        if len(block) != scene_count:
            raise RuntimeError(
                f"rank {rank}: expected {scene_count} reviewed candidates, got {len(block)}"
            )
        increments = {
            status: int(block["status"].eq(status).sum()) for status in STATUS_ORDER
        }
        for status in STATUS_ORDER:
            cumulative[status] += increments[status]
        reviewed_candidates = scene_count * rank
        if sum(cumulative.values()) != reviewed_candidates:
            raise RuntimeError(
                f"rank {rank}: cumulative status counts do not equal review budget"
            )
        rows.append(
            {
                "top_n_per_scene": rank,
                "candidates_reviewed": reviewed_candidates,
                "new_confirmed_collisions": increments["collision"],
                "cumulative_confirmed_collisions": cumulative["collision"],
                "cumulative_non_collisions": cumulative["non_collision"],
                "cumulative_ambiguous": cumulative["ambiguous"],
                "top_n_precision": cumulative["collision"] / reviewed_candidates,
            }
        )
    return pd.DataFrame(rows)
