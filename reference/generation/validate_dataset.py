#!/usr/bin/env python3
"""Validate the non-negotiable collision-benchmark generation rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import generate_clean as base


DATLEN = 1024
PROPAGATION_DATLEN = 1536
DX_UM = 10.0
MARGIN_UM = (PROPAGATION_DATLEN - DATLEN) * DX_UM / 2.0
WIDTH_UM = DATLEN * DX_UM
HEIGHT_UM = DATLEN * DX_UM
DEPTH_UM = 1024 * 100.0
def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def outside_propagation_xy(x_um: float, y_um: float) -> bool:
    return (
        x_um < -MARGIN_UM
        or x_um >= WIDTH_UM + MARGIN_UM
        or y_um < -MARGIN_UM
        or y_um >= HEIGHT_UM + MARGIN_UM
    )


def inside_propagation_domain(x_um: float, y_um: float, z_um: float) -> bool:
    return (
        -MARGIN_UM <= x_um < WIDTH_UM + MARGIN_UM
        and -MARGIN_UM <= y_um < HEIGHT_UM + MARGIN_UM
        and 0.0 <= z_um < DEPTH_UM
    )


def inside_nominal_volume(x_um: float, y_um: float, z_um: float) -> bool:
    return (
        0.0 <= x_um < WIDTH_UM
        and 0.0 <= y_um < HEIGHT_UM
        and 0.0 <= z_um < DEPTH_UM
    )


def full_axis_step_interval(
    position: float,
    step: float,
    lower: float,
    upper: float,
) -> tuple[int, int | float] | None:
    if step > 0.0:
        first = max(0, math.ceil((lower - position) / step))
        last: int | float = math.ceil((upper - position) / step) - 1
    elif step < 0.0:
        first = max(0, math.floor((upper - position) / step) + 1)
        last = math.floor((lower - position) / step)
    elif lower <= position < upper:
        first, last = 0, math.inf
    else:
        return None
    return (first, last) if first <= last else None


def complete_xy_residence_is_z_safe(
    row: dict[str, str],
    fps: float,
    safety_margin_um: float,
) -> bool:
    x_interval = full_axis_step_interval(
        as_float(row, "initial_x_um"),
        as_float(row, "initial_vx_um_s") / fps,
        -MARGIN_UM,
        WIDTH_UM + MARGIN_UM,
    )
    y_interval = full_axis_step_interval(
        as_float(row, "initial_y_um"),
        as_float(row, "initial_vy_um_s") / fps,
        -MARGIN_UM,
        HEIGHT_UM + MARGIN_UM,
    )
    if x_interval is None or y_interval is None:
        return True
    first = max(x_interval[0], y_interval[0])
    last = min(x_interval[1], y_interval[1])
    if first > last:
        return True
    z_step = as_float(row, "initial_vz_um_s") / fps
    z_first = as_float(row, "initial_z_um") + z_step * first
    if math.isinf(last):
        return (
            abs(z_step) <= 1e-15
            and safety_margin_um <= z_first < DEPTH_UM - safety_margin_um
        )
    z_last = as_float(row, "initial_z_um") + z_step * last
    return (
        min(z_first, z_last) >= safety_margin_um
        and max(z_first, z_last) < DEPTH_UM - safety_margin_um
    )


def validate(dataset: Path, labels_only: bool = False) -> dict[str, object]:
    particles_path = dataset / "labels" / "particles.csv"
    initial_particles_path = dataset / "labels" / "particles_initial.csv"
    trajectories_path = dataset / "labels" / "trajectories.csv"
    collisions_path = dataset / "labels" / "collision_events.csv"
    parameters_path = dataset / "metadata" / "parameters.json"
    for path in [
        particles_path,
        initial_particles_path,
        trajectories_path,
        collisions_path,
        parameters_path,
    ]:
        if not path.is_file():
            raise AssertionError(f"missing required file: {path}")

    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    assert parameters["datlen"] == DATLEN
    assert parameters["propagation_margin_px"] == 256.0
    assert parameters["c002_front_distance_um"] == 74900.0
    assert parameters["c001_front_distance_um"] == 96350.0
    assert parameters["noise_enabled"] is False
    assert parameters["noise_model"] == "none"
    expected_frames = int(parameters["frames"])
    expected_collisions = int(parameters["target_planned_collision_count"])
    target_nominal_particles = int(
        parameters["nominal_observation_particles_per_frame_target"]
    )
    fps = float(parameters["fps"])
    z_safety_margin_um = float(parameters["z_boundary_safety_margin_um"])
    y_velocity_offset_um_s = float(parameters["y_velocity_offset_um_s"])
    small_y_jitter_enabled = bool(
        parameters.get("small_particle_y_velocity_jitter_enabled", False)
    )
    small_y_jitter_max_diameter_um = float(
        parameters.get(
            "small_particle_y_velocity_jitter_max_diameter_um",
            150.0,
        )
    )
    small_y_jitter_min_fraction = float(
        parameters.get("small_particle_y_velocity_jitter_min_fraction", -0.10)
    )
    small_y_jitter_max_fraction = float(
        parameters.get("small_particle_y_velocity_jitter_max_fraction", 0.10)
    )
    assert parameters["initial_particle_spatial_distribution"] == (
        "uniform_center_coordinates_in_nominal_observation_volume_at_frame_0_"
        "with_overlap_rejection"
    )
    assert parameters["z_boundary_policy"] == (
        "sample only trajectories whose z center remains inside 0 <= z < depth_um "
        "for the complete residence through the 1536x1536 XY propagation plane, "
        "including extrapolation beyond the recording end; never create or remove "
        "a visible particle because of a z-boundary crossing"
    )

    frame_counts: Counter[int] = Counter()
    nominal_frame_counts: Counter[int] = Counter()
    initial_frame_nominal_count = 0
    particle_rows: list[dict[str, str]] = []
    with particles_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            x_um = as_float(row, "x_um")
            y_um = as_float(row, "y_um")
            z_um = as_float(row, "z_um")
            assert -MARGIN_UM <= x_um < WIDTH_UM + MARGIN_UM
            assert -MARGIN_UM <= y_um < HEIGHT_UM + MARGIN_UM
            assert 0.0 <= z_um < DEPTH_UM
            frame_counts[frame] += 1
            if inside_nominal_volume(x_um, y_um, z_um):
                nominal_frame_counts[frame] += 1
                if frame == 0:
                    initial_frame_nominal_count += 1
            particle_rows.append(row)
    assert set(frame_counts) == set(range(expected_frames))
    assert set(nominal_frame_counts) == set(range(expected_frames))
    assert set(nominal_frame_counts.values()) == {target_nominal_particles}
    assert initial_frame_nominal_count == target_nominal_particles

    initial_rows = []
    with initial_particles_path.open(newline="") as f:
        for row in csv.DictReader(f):
            assert row["birth_reason"] == "initial"
            assert row["entry_face"] == "initial_uniform_nominal_volume"
            assert int(row["spawn_outside_volume"]) == 0
            assert inside_nominal_volume(
                as_float(row, "x_um"),
                as_float(row, "y_um"),
                as_float(row, "z_um"),
            )
            initial_rows.append(row)
    assert len(initial_rows) == target_nominal_particles

    checked_births = 0
    checked_initial_births = 0
    checked_xy_boundary_exits = 0
    z_boundary_disappearances = 0
    checked_simulation_end_full_xy_residence = 0
    y_jitter_eligible_particles = 0
    y_jitter_varied_particles = 0
    y_jitter_relative_changes: list[float] = []
    noneligible_y_baseline_max_abs_error_um_s = 0.0
    trajectories: dict[int, dict[str, str]] = {}
    with trajectories_path.open(newline="") as f:
        for row in csv.DictReader(f):
            trajectories[int(row["particle_id"])] = row
            birth_reason = row["birth_reason"]
            x_um = as_float(row, "initial_x_um")
            y_um = as_float(row, "initial_y_um")
            z_um = as_float(row, "initial_z_um")
            if birth_reason == "initial":
                assert int(row["birth_frame"]) == 0
                assert inside_nominal_volume(x_um, y_um, z_um)
                checked_initial_births += 1
            elif birth_reason != "collision_merge":
                assert outside_propagation_xy(x_um, y_um)
                checked_births += 1

            death_reason = row["death_reason"]
            if death_reason == "out_of_propagation_domain":
                z_boundary_disappearances += 1
            if death_reason == "out_of_xy_propagation_domain":
                final_x_um = as_float(row, "final_x_um")
                final_y_um = as_float(row, "final_y_um")
                assert outside_propagation_xy(final_x_um, final_y_um)
                checked_xy_boundary_exits += 1
            if death_reason == "simulation_end":
                assert complete_xy_residence_is_z_safe(
                    row,
                    fps,
                    z_safety_margin_um,
                )
                checked_simulation_end_full_xy_residence += 1

            if birth_reason != "collision_merge":
                diameter_um = as_float(row, "initial_diameter_um")
                actual_vy_um_s = as_float(row, "initial_vy_um_s")
                baseline_vy_um_s = (
                    base.terminal_velocity_beard_1976_um_s(diameter_um)
                    + y_velocity_offset_um_s
                )
                if (
                    small_y_jitter_enabled
                    and diameter_um <= small_y_jitter_max_diameter_um
                ):
                    y_jitter_eligible_particles += 1
                    if math.isclose(baseline_vy_um_s, 0.0, abs_tol=1e-12):
                        assert math.isclose(
                            actual_vy_um_s,
                            0.0,
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                        relative_change = 0.0
                    else:
                        relative_change = actual_vy_um_s / baseline_vy_um_s - 1.0
                    assert (
                        small_y_jitter_min_fraction - 1e-12
                        <= relative_change
                        <= small_y_jitter_max_fraction + 1e-12
                    )
                    y_jitter_relative_changes.append(relative_change)
                    if not math.isclose(
                        actual_vy_um_s,
                        baseline_vy_um_s,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        y_jitter_varied_particles += 1
                else:
                    error_um_s = abs(actual_vy_um_s - baseline_vy_um_s)
                    noneligible_y_baseline_max_abs_error_um_s = max(
                        noneligible_y_baseline_max_abs_error_um_s,
                        error_um_s,
                    )
                    assert math.isclose(
                        actual_vy_um_s,
                        baseline_vy_um_s,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
    assert checked_initial_births == target_nominal_particles
    assert z_boundary_disappearances == 0
    if small_y_jitter_enabled:
        assert y_jitter_eligible_particles > 0
        assert y_jitter_varied_particles > 0

    checked_linear_label_rows = 0
    for row in particle_rows:
        trajectory = trajectories[int(row["particle_id"])]
        steps = int(row["frame"]) - int(trajectory["birth_frame"])
        assert steps >= 0
        for coordinate, velocity in (
            ("x_um", "initial_vx_um_s"),
            ("y_um", "initial_vy_um_s"),
            ("z_um", "initial_vz_um_s"),
        ):
            expected = (
                as_float(trajectory, f"initial_{coordinate}")
                + as_float(trajectory, velocity) / fps * steps
            )
            assert math.isclose(
                as_float(row, coordinate),
                expected,
                rel_tol=0.0,
                abs_tol=2e-4,
            )
        for velocity in ("vx_um_s", "vy_um_s", "vz_um_s"):
            assert math.isclose(
                as_float(row, velocity),
                as_float(trajectory, f"initial_{velocity}"),
                rel_tol=0.0,
                abs_tol=2e-4,
            )
        checked_linear_label_rows += 1

    collision_count = 0
    with collisions_path.open(newline="") as f:
        for row in csv.DictReader(f):
            x_um = as_float(row, "x_um")
            y_um = as_float(row, "y_um")
            z_um = as_float(row, "z_um")
            assert 0.0 <= x_um < WIDTH_UM
            assert 0.0 <= y_um < HEIGHT_UM
            assert 0.0 <= z_um < DEPTH_UM
            collision_count += 1
    assert collision_count == expected_collisions

    camera_counts = {}
    if not labels_only:
        for camera in ("C001", "C002"):
            image_dir = dataset / "01_bkgrem" / f"{camera}H001S0011"
            camera_counts[camera] = len(list(image_dir.glob("*.png")))
            assert camera_counts[camera] == expected_frames

    minip_summary_path = dataset / "metadata" / "minip_summary.json"
    minip_count: int | None = None
    if not labels_only and minip_summary_path.is_file():
        minip_summary = json.loads(minip_summary_path.read_text(encoding="utf-8"))
        minip_dir = dataset / "05_03_xyproj" / "C000H001S0011"
        minip_count = len(list(minip_dir.glob("*.png")))
        assert minip_count == expected_frames
        assert int(minip_summary["images"]) == expected_frames

    counts = list(frame_counts.values())
    nominal_counts = [
        nominal_frame_counts.get(frame, 0)
        for frame in range(expected_frames)
    ]
    result = {
        "status": "PASS",
        "frames": len(frame_counts),
        "rendered_particles_per_frame": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
        },
        "nominal_observation_particles_per_frame": {
            "target": target_nominal_particles,
            "min": min(nominal_counts),
            "max": max(nominal_counts),
            "mean": sum(nominal_counts) / len(nominal_counts),
        },
        "initial_nominal_observation_particles": initial_frame_nominal_count,
        "particles_initial_rows": len(initial_rows),
        "checked_uniform_initial_births": checked_initial_births,
        "checked_outside_births": checked_births,
        "checked_linear_label_rows": checked_linear_label_rows,
        "checked_xy_boundary_exits": checked_xy_boundary_exits,
        "checked_simulation_end_full_xy_residence": (
            checked_simulation_end_full_xy_residence
        ),
        "small_particle_y_velocity_jitter": {
            "enabled": small_y_jitter_enabled,
            "maximum_diameter_um": small_y_jitter_max_diameter_um,
            "eligible_nonmerge_particles": y_jitter_eligible_particles,
            "varied_nonmerge_particles": y_jitter_varied_particles,
            "relative_change_min": (
                min(y_jitter_relative_changes)
                if y_jitter_relative_changes
                else None
            ),
            "relative_change_max": (
                max(y_jitter_relative_changes)
                if y_jitter_relative_changes
                else None
            ),
            "noneligible_baseline_max_abs_error_um_s": (
                noneligible_y_baseline_max_abs_error_um_s
            ),
            "constant_velocity_rows_checked": checked_linear_label_rows,
        },
        "z_boundary_disappearances": z_boundary_disappearances,
        "positive_collisions_inside_nominal_volume": collision_count,
        "camera_png_counts": camera_counts,
        "minip_png_count": minip_count,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Validate labels and metadata before the expensive image render.",
    )
    args = parser.parse_args()
    result = validate(args.dataset_dir.resolve(), labels_only=args.labels_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
