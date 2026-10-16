#!/usr/bin/env python3
"""Audit unintended overlaps between parents of different planned events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OVERLAP_TOLERANCE_UM = 1e-6
MAX_EXAMPLES = 20


def planned_parent_windows(
    dataset: Path,
) -> tuple[dict[int, dict[str, int | str]], int]:
    design = json.loads(
        (dataset / "metadata" / "half_mixture_design.json").read_text(
            encoding="utf-8"
        )
    )
    trajectories = pd.read_csv(
        dataset / "labels" / "trajectories.csv",
        usecols=["particle_id", "birth_frame", "birth_reason"],
    ).set_index("particle_id")
    windows: dict[int, dict[str, int | str]] = {}
    event_count = 0
    for plan in design["plans"]:
        event_id = int(plan["planned_event_id"])
        component = str(plan["sampling_component"])
        activation_frame = int(plan["activation_frame"])
        collision_frame = int(plan["collision_frame"])
        event_count += 1
        for slot in (1, 2):
            particle_id = int(plan[f"parent{slot}_id"])
            trajectory = trajectories.loc[particle_id]
            birth_reason = str(trajectory["birth_reason"])
            if (
                component == "coverage_uniform"
                or birth_reason
                in {
                    "planned_collision_parent",
                    "coverage_planned_collision_parent",
                }
            ):
                start_frame = int(trajectory["birth_frame"])
            else:
                # A legacy target is an ordinary background particle until the
                # generator selects it at activation_frame.
                start_frame = activation_frame
            windows[particle_id] = {
                "planned_event_id": event_id,
                "sampling_component": component,
                "parent_slot": slot,
                "start_frame": start_frame,
                "end_frame_exclusive": collision_frame,
            }
    return windows, event_count


def audit(dataset: Path) -> dict[str, Any]:
    windows, event_count = planned_parent_windows(dataset)
    usecols = [
        "frame",
        "particle_id",
        "x_um",
        "y_um",
        "z_um",
        "diameter_um",
    ]
    particles = pd.read_csv(
        dataset / "labels" / "particles.csv",
        usecols=usecols,
        dtype={
            "frame": np.int32,
            "particle_id": np.int32,
            "x_um": np.float64,
            "y_um": np.float64,
            "z_um": np.float64,
            "diameter_um": np.float64,
        },
    )
    window_frame = pd.Series(
        {
            particle_id: int(window["start_frame"])
            for particle_id, window in windows.items()
        }
    )
    window_end = pd.Series(
        {
            particle_id: int(window["end_frame_exclusive"])
            for particle_id, window in windows.items()
        }
    )
    event_ids = pd.Series(
        {
            particle_id: int(window["planned_event_id"])
            for particle_id, window in windows.items()
        }
    )
    selected = particles[particles["particle_id"].isin(windows)].copy()
    selected["audit_start_frame"] = selected["particle_id"].map(window_frame)
    selected["audit_end_frame"] = selected["particle_id"].map(window_end)
    selected = selected[
        selected["frame"].ge(selected["audit_start_frame"])
        & selected["frame"].lt(selected["audit_end_frame"])
    ].copy()
    selected["planned_event_id"] = (
        selected["particle_id"].map(event_ids).astype(np.int32)
    )

    duplicate_rows = int(
        selected.duplicated(subset=["frame", "particle_id"]).sum()
    )
    if duplicate_rows:
        raise AssertionError(
            "planned-parent audit found duplicate particle/frame rows: "
            f"{duplicate_rows}"
        )

    overlapping_pair_frames = 0
    overlap_frames: set[int] = set()
    overlap_particle_pairs: set[tuple[int, int]] = set()
    overlap_event_pairs: set[tuple[int, int]] = set()
    minimum_surface_gap_um = float("inf")
    closest: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []
    compared_pair_frames = 0

    for frame, rows in selected.groupby("frame", sort=True):
        if len(rows) < 2:
            continue
        particle_ids = rows["particle_id"].to_numpy(dtype=np.int64)
        planned_event_ids = rows["planned_event_id"].to_numpy(dtype=np.int64)
        coordinates = rows[["x_um", "y_um", "z_um"]].to_numpy(
            dtype=np.float64
        )
        radii = 0.5 * rows["diameter_um"].to_numpy(dtype=np.float64)
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        distance_squared = np.einsum(
            "ijk,ijk->ij",
            deltas,
            deltas,
            optimize=True,
        )
        radius_sums = radii[:, None] + radii[None, :]
        upper = np.triu(
            np.ones((len(rows), len(rows)), dtype=bool),
            k=1,
        )
        eligible = upper & (
            planned_event_ids[:, None] != planned_event_ids[None, :]
        )
        left, right = np.nonzero(eligible)
        if left.size == 0:
            continue
        compared_pair_frames += int(left.size)
        distances = np.sqrt(distance_squared[left, right])
        gaps = distances - radius_sums[left, right]
        closest_index = int(np.argmin(gaps))
        frame_minimum = float(gaps[closest_index])
        if frame_minimum < minimum_surface_gap_um:
            minimum_surface_gap_um = frame_minimum
            i = int(left[closest_index])
            j = int(right[closest_index])
            closest = {
                "frame": int(frame),
                "particle1_id": int(particle_ids[i]),
                "particle2_id": int(particle_ids[j]),
                "planned_event1_id": int(planned_event_ids[i]),
                "planned_event2_id": int(planned_event_ids[j]),
                "center_distance_um": float(distances[closest_index]),
                "radius_sum_um": float(radius_sums[i, j]),
                "surface_gap_um": frame_minimum,
            }

        overlap_indices = np.flatnonzero(gaps < -OVERLAP_TOLERANCE_UM)
        if overlap_indices.size == 0:
            continue
        overlap_frames.add(int(frame))
        overlapping_pair_frames += int(overlap_indices.size)
        for overlap_index in overlap_indices:
            i = int(left[overlap_index])
            j = int(right[overlap_index])
            particle_pair = tuple(
                sorted((int(particle_ids[i]), int(particle_ids[j])))
            )
            event_pair = tuple(
                sorted(
                    (
                        int(planned_event_ids[i]),
                        int(planned_event_ids[j]),
                    )
                )
            )
            overlap_particle_pairs.add(particle_pair)
            overlap_event_pairs.add(event_pair)
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "frame": int(frame),
                        "particle1_id": int(particle_ids[i]),
                        "particle2_id": int(particle_ids[j]),
                        "planned_event1_id": int(planned_event_ids[i]),
                        "planned_event2_id": int(planned_event_ids[j]),
                        "center_distance_um": float(
                            distances[overlap_index]
                        ),
                        "radius_sum_um": float(radius_sums[i, j]),
                        "penetration_um": float(-gaps[overlap_index]),
                    }
                )

    return {
        "status": "PASS" if overlapping_pair_frames == 0 else "FAIL",
        "definition": (
            "Parent states belonging to different planned events are compared "
            "for every simultaneously recorded frame in their planned-parent "
            "windows; intended parents from the same event are excluded."
        ),
        "tolerance_um": OVERLAP_TOLERANCE_UM,
        "planned_event_count": event_count,
        "planned_parent_count": len(windows),
        "planned_parent_state_rows": int(len(selected)),
        "compared_pair_frames": compared_pair_frames,
        "overlapping_pair_frames": overlapping_pair_frames,
        "overlap_frame_count": len(overlap_frames),
        "overlap_particle_pair_count": len(overlap_particle_pairs),
        "overlap_event_pair_count": len(overlap_event_pairs),
        "minimum_surface_gap_um": (
            minimum_surface_gap_um
            if np.isfinite(minimum_surface_gap_um)
            else None
        ),
        "closest_pair_frame": closest,
        "overlap_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the JSON result to this path.",
    )
    args = parser.parse_args()
    result = audit(args.dataset_dir.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
