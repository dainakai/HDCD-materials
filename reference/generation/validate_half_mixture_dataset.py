#!/usr/bin/env python3
"""Validate collision-benchmark-half-mixture/v1 labels and image products."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import audit_planned_parent_overlaps
import half_mixture
import validate_dataset


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def validate_half_mixture(
    dataset: Path,
    *,
    labels_only: bool,
    allow_missing_minip: bool,
) -> dict[str, Any]:
    base_result = validate_dataset.validate(dataset, labels_only=labels_only)
    parameters = json.loads(
        (dataset / "metadata" / "parameters.json").read_text(encoding="utf-8")
    )
    generated = json.loads(
        (dataset / "metadata" / "generated_summary.json").read_text(
            encoding="utf-8"
        )
    )
    design = json.loads(
        (dataset / "metadata" / "half_mixture_design.json").read_text(
            encoding="utf-8"
        )
    )
    events = read_csv(dataset / "labels" / "collision_events.csv")
    trajectories = {
        as_int(row, "particle_id"): row
        for row in read_csv(dataset / "labels" / "trajectories.csv")
    }

    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        if not name.startswith("event_"):
            checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    check(
        "format_version",
        parameters.get("dataset_format_version")
        == half_mixture.FORMAT_VERSION
        and design.get("format_version") == half_mixture.FORMAT_VERSION,
    )
    check("half_mixture_enabled", parameters["half_mixture_25um_enabled"])
    check(
        "all_events_are_planned",
        all(row["event_type"] == "planned" for row in events),
    )
    check(
        "planned_event_count",
        len(events) == int(parameters["target_planned_collision_count"]),
    )
    check(
        "no_unscheduled_events",
        int(generated["unscheduled_planned_collision_count"]) == 0,
    )
    check(
        "no_missed_events",
        int(generated["missed_planned_collision_count"]) == 0,
    )

    component_counts = Counter(row["sampling_component"] for row in events)
    expected_coverage = int(design["coverage_target_count"])
    expected_legacy = int(design["legacy_target_count"])
    check(
        "component_counts",
        component_counts["coverage_uniform"] == expected_coverage
        and component_counts["legacy_chase"] == expected_legacy,
    )
    check(
        "planned_event_ids_unique",
        len({as_int(row, "planned_event_id") for row in events})
        == len(events),
    )

    edges = [float(value) for value in design["diameter_bin_edges_um"]]
    expected_allocations = Counter(
        (
            int(row["pair_low"]),
            int(row["pair_high"]),
            int(row["speed_tier"]) + 1,
            int(row["impact_tier"]) + 1,
        )
        for row in design["coverage_allocations"]
    )
    actual_allocations: Counter[tuple[int, int, int, int]] = Counter()
    coverage_pair_counts: Counter[tuple[int, int]] = Counter()
    mixed_pair_counts: Counter[tuple[int, int]] = Counter()
    maximum_volume_relative_error = 0.0
    maximum_momentum_relative_error = 0.0
    for row in events:
        pair = (
            as_int(row, "pair_low_25um"),
            as_int(row, "pair_high_25um"),
        )
        mixed_pair_counts[pair] += 1
        d1 = as_float(row, "parent1_diameter_um")
        d2 = as_float(row, "parent2_diameter_um")
        expected_pair = tuple(
            sorted(
                (
                    half_mixture.diameter_bin_index(
                        d1,
                        np.asarray(edges),
                    ),
                    half_mixture.diameter_bin_index(
                        d2,
                        np.asarray(edges),
                    ),
                )
            )
        )
        check(
            f"event_{row['collision_event_id']}_25um_pair",
            pair == expected_pair,
        )
        impact = as_float(row, "normalized_impact_parameter")
        percentile = as_float(row, "conditional_speed_percentile")
        speed_tier = as_int(row, "relative_speed_tier")
        impact_tier = as_int(row, "impact_parameter_tier")
        check(
            f"event_{row['collision_event_id']}_speed_tier",
            speed_tier
            == min(
                int(percentile * half_mixture.SPEED_TIER_COUNT),
                half_mixture.SPEED_TIER_COUNT - 1,
            )
            + 1,
        )
        check(
            f"event_{row['collision_event_id']}_impact_tier",
            impact_tier
            == min(
                int(min(impact, 0.999999999) * half_mixture.IMPACT_TIER_COUNT),
                half_mixture.IMPACT_TIER_COUNT - 1,
            )
            + 1,
        )
        check(
            f"event_{row['collision_event_id']}_impact_range",
            0.0 <= impact < 1.0,
        )
        check(
            f"event_{row['collision_event_id']}_overlap",
            as_float(row, "distance_um") < as_float(row, "radius_sum_um"),
        )

        child_diameter = as_float(row, "child_diameter_um")
        volume_error = abs(
            child_diameter**3 - (d1**3 + d2**3)
        ) / (d1**3 + d2**3)
        maximum_volume_relative_error = max(
            maximum_volume_relative_error,
            volume_error,
        )
        mass1 = d1**3
        mass2 = d2**3
        for axis in ("x", "y", "z"):
            parent_momentum = (
                mass1 * as_float(row, f"parent1_v{axis}_um_s")
                + mass2 * as_float(row, f"parent2_v{axis}_um_s")
            )
            child_momentum = (
                mass1 + mass2
            ) * as_float(row, f"child_v{axis}_um_s")
            scale = max(abs(parent_momentum), abs(child_momentum), 1.0)
            maximum_momentum_relative_error = max(
                maximum_momentum_relative_error,
                abs(parent_momentum - child_momentum) / scale,
            )

        if row["sampling_component"] == "coverage_uniform":
            check(
                f"event_{row['collision_event_id']}_approaching",
                as_float(row, "approach_dot_um2_s") < 0.0,
            )
            coverage_pair_counts[pair] += 1
            actual_allocations[
                (pair[0], pair[1], speed_tier, impact_tier)
            ] += 1
            parent1 = trajectories[as_int(row, "parent1_id")]
            parent2 = trajectories[as_int(row, "parent2_id")]
            check(
                f"event_{row['collision_event_id']}_coverage_parent_birth_reason",
                parent1["birth_reason"]
                == "coverage_planned_collision_parent"
                and parent2["birth_reason"]
                == "coverage_planned_collision_parent",
            )
            check(
                f"event_{row['collision_event_id']}_coverage_parent_lead",
                as_int(row, "frame") - as_int(parent1, "birth_frame")
                == int(design["coverage_parent_lead_frames"])
                and as_int(row, "frame") - as_int(parent2, "birth_frame")
                == int(design["coverage_parent_lead_frames"]),
            )

    checks["all_event_geometry_and_tier_checks"] = True
    checks["all_coverage_parent_lineage_checks"] = True
    check(
        "coverage_allocation_matches_design",
        actual_allocations == expected_allocations,
    )
    check(
        "volume_conservation",
        maximum_volume_relative_error <= 1e-12,
    )
    check(
        "momentum_conservation",
        maximum_momentum_relative_error <= 1e-12,
    )

    if expected_coverage == 1_000 and len(edges) == 12:
        check("coverage_all_66_pairs", len(coverage_pair_counts) == 66)
        check(
            "coverage_pair_counts_15_or_16",
            set(coverage_pair_counts.values()) == {15, 16},
        )
        check(
            "mixed_pair_minimum_15",
            len(mixed_pair_counts) == 66
            and min(mixed_pair_counts.values()) >= 15,
        )
        occupied_per_pair = Counter()
        for low, high, speed_tier, impact_tier in actual_allocations:
            if actual_allocations[(low, high, speed_tier, impact_tier)] > 0:
                occupied_per_pair[(low, high)] += 1
        check(
            "coverage_all_15_joint_strata",
            len(occupied_per_pair) == 66
            and set(occupied_per_pair.values()) == {15},
        )

    overlap_audit = audit_planned_parent_overlaps.audit(dataset)
    (dataset / "metadata" / "planned_parent_overlap_audit.json").write_text(
        json.dumps(overlap_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    check(
        "planned_parent_no_unintended_overlaps",
        overlap_audit["status"] == "PASS",
    )

    raw_counts: dict[str, int] = {}
    minip_count: int | None = None
    if not labels_only:
        expected_frames = int(parameters["frames"])
        for camera in ("C001", "C002"):
            raw_dir = dataset / "01_bkgrem" / f"{camera}H001S0011"
            raw_counts[camera] = len(list(raw_dir.glob("*.png")))
        check(
            "raw_image_counts",
            raw_counts == {
                "C001": expected_frames,
                "C002": expected_frames,
            },
        )
        minip_dir = dataset / "05_03_xyproj" / "C000H001S0011"
        minip_count = len(list(minip_dir.glob("*.png")))
        if not allow_missing_minip:
            check("minip_image_count", minip_count == expected_frames)
            check(
                "minip_summary_exists",
                (dataset / "metadata" / "minip_summary.json").is_file(),
            )

    return {
        "status": "PASS",
        "format_version": half_mixture.FORMAT_VERSION,
        "checks": checks,
        "base_validation": base_result,
        "event_counts": {
            "total": len(events),
            "coverage_uniform": component_counts["coverage_uniform"],
            "legacy_chase": component_counts["legacy_chase"],
        },
        "coverage": {
            "nonempty_pair_count": len(coverage_pair_counts),
            "pair_count_min": (
                min(coverage_pair_counts.values())
                if coverage_pair_counts
                else 0
            ),
            "pair_count_max": (
                max(coverage_pair_counts.values())
                if coverage_pair_counts
                else 0
            ),
            "occupied_joint_strata": len(actual_allocations),
        },
        "conservation": {
            "maximum_volume_relative_error": maximum_volume_relative_error,
            "maximum_momentum_relative_error": maximum_momentum_relative_error,
        },
        "planned_parent_overlap_audit": overlap_audit,
        "raw_png_counts": raw_counts,
        "minip_png_count": minip_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument(
        "--allow-missing-minip",
        action="store_true",
        help="Validate labels and raw images without requiring MinIP output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the JSON result to this path.",
    )
    args = parser.parse_args()
    result = validate_half_mixture(
        args.dataset_dir.resolve(),
        labels_only=args.labels_only,
        allow_missing_minip=args.allow_missing_minip,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
