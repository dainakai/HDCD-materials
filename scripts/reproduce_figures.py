#!/usr/bin/env python3
"""Recalculate and render Figures 4, 5, 7, 8 and 11–15 from bundled numerical inputs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import numpy as np
import pandas as pd
import polars as pl

from ra3d.evaluation import event_curve
from ra3d._core.matching import candidate_truth_adjacency
from figure_support.fonts import register_fonts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper_figures"
REFERENCE = DATA / "reference"
FIGURES = [4, 5, 7, 8, 11, 12, 13, 14, 15]
PLOT_PYTHON = ROOT / ".venv-figures311/bin/python"


def plot_311(number, output):
    subprocess.run([str(PLOT_PYTHON), str(ROOT / "scripts/render_figures_311.py"),
                    "--figure", str(number), "--output", str(output)], check=True)


def table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")


def check(actual, expected, name, *, rtol=1e-10, atol=1e-9):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True, err_msg=name)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fig04(output):
    from figure_support import population_size as size, population_count as count

    bins, curves = size.load_scene_curves(DATA / "fig04/diameter_by_scene.csv")
    mean = curves.mean(axis=0)
    low, high = size.bootstrap_mean_interval(curves, replicates=50000, seed=260715, chunk_size=1000)
    expected = table(REFERENCE / "diameter_mean_ci.csv")
    check(np.c_[mean, low, high], expected.iloc[:, 3:].to_numpy(), "Figure 4(a) mean and CI")
    summary = bins.assign(mean=mean, ci025=low, ci975=high)
    summary.to_csv(output / "diameter_mean_ci.csv", index=False)
    size.plot_distribution(output / "particle_size_distribution_mean_ci_260715",
                           bins.diameter_center_um.to_numpy(), mean, low, high, plot_min=25, plot_max=300)
    frames = pl.read_parquet(DATA / "fig04/frame_metrics.parquet")
    scenes = sorted(frames["scene"].unique().to_list())
    summaries = count.scene_summary(frames, scenes)
    original = table(REFERENCE / "scene_statistics.csv")
    columns = ["particles_mean", "particles_std", "lwc_g_m3_mean", "lwc_g_m3_std"]
    check(summaries.select(columns).to_numpy(), original[columns].to_numpy(), "Figure 4(b) means and SDs")
    summaries.write_csv(output / "scene_statistics.csv")
    # The original two panels were created by separate programs with separate rc defaults.
    mpl.rcdefaults()
    count.OUTPUT_DIR = output
    count.style_publication()
    count.render_scene_summary(summaries)
    return {"scenes": len(scenes), "frame_rows": frames.height, "bootstrap_replicates": 50000,
            "bootstrap_seed": 260715, "reference_tables_match": True}


def fig05(output):
    from figure_support import velocity, beard1976

    tracks = table(DATA / "fig05/velocity_tracks.parquet")
    assert tracks.scene.nunique() == 14
    velocity.plot_velocity(tracks, beard1976.terminal_velocity_beard_1976_um_s,
                           output / "velocity_distributions_260715.pdf",
                           output / "velocity_distributions_260715.png")
    return {"tracks": len(tracks), "scenes": 14, "diameter_um": [25, 300], "minimum_valid_points": 5,
            "upward_offset_m_s": 0.26}


def fig07(output):
    from figure_support import particle_metrics as metrics, particle_plot as plot
    register_fonts("1.07.4")

    metrics.TRUTH_PATH = ROOT / "data/labels/validation/seed260745/particles.parquet"
    metrics.LOW_CONF_DETECTIONS = DATA / "fig07/low_confidence_detections.parquet"
    truth = metrics.load_truth()
    detections, flags = metrics.greedy_detection_flags(truth)
    curve = metrics.precision_recall(detections.conf.to_numpy(), flags, 1200000)
    tracks = table(DATA / "fig07/paired_tracks_20mm.parquet")
    errors = metrics.match_tracks(truth, {name: tracks[name].to_numpy() for name in tracks})
    xy = plot.finite(errors["xy_um"])
    z_um = plot.finite(np.abs(errors["z_um"]))
    relative = plot.finite(errors["diameter_relative"])
    saved = json.loads((REFERENCE / "particle_metrics_20mm.json").read_text())
    check(curve["average_precision"], saved["detection"]["average_precision"], "Figure 7(a) AP")
    for values, key in [(xy, "xy_error_smoothed_um"), (z_um, "z_absolute_error_smoothed_um")]:
        check(np.percentile(values, [50, 90, 95]), [saved[key][k] for k in ["median", "p90", "p95"]], key)
    check(np.median(relative), saved["diameter_relative_error"]["median"], "Figure 7(d) median")
    assert len(xy) == saved["trajectory_matching"]["matched_points"]
    plot.OUTPUT_DIR = output
    plot.configure_plotting()
    plot.plot_pr(curve)
    upper = max(40, math.ceil(float(np.percentile(xy, 99)) / 10) * 10)
    plot.plot_positive_histogram(xy, np.percentile(xy, [50, 90, 95]), xlim=(0, upper), bins=140,
        xlabel=r"In-plane center error, $\sqrt{\Delta x^2+\Delta y^2}$ [$\mu$m]", unit=r"$\mu$m",
        facecolor="#4C78A8", edgecolor="#2F4B66", filename="xy_center_error_smoothed_hist_seed260745.pdf",
        decimals=2, major_tick=10)
    plot.plot_positive_histogram(z_um / 1000, np.percentile(z_um / 1000, [50, 90, 95]), xlim=(0, 5), bins=125,
        xlabel=r"Axial absolute error, $|\Delta z|$ [mm]", unit="mm", facecolor="#568B7B", edgecolor="#315E53",
        filename="axial_error_smoothed_hist_seed260745.pdf", decimals=3, major_tick=1)
    plot.plot_diameter_histogram(100 * relative, np.percentile(100 * relative, [2.5, 5, 50, 95, 97.5]))
    for prefix, name in enumerate(["particle_detection_pr_curve_seed260745", "xy_center_error_smoothed_hist_seed260745",
                                   "axial_error_smoothed_hist_seed260745", "diameter_relative_error_hist_seed260745"], 1):
        (output / f"{name}.pdf").rename(output / f"{prefix:02d}_{name}.pdf")
    pd.DataFrame({key: curve[key] for key in ["confidence", "recall", "precision", "f1"]}).to_parquet(
        output / "particle_pr_curve.parquet", index=False, compression="zstd")
    pd.DataFrame({key: errors[key] for key in ["xy_um", "z_um", "diameter_relative"]}).to_parquet(
        output / "matched_particle_errors.parquet", index=False, compression="zstd")
    plot_311(7, output)
    return {"particle_ap": curve["average_precision"], "low_confidence_detections": len(detections),
            "matched_trajectory_points": len(xy), "xy_percentiles_um": np.percentile(xy, [50, 90, 95]).tolist(),
            "z_absolute_percentiles_mm": np.percentile(z_um / 1000, [50, 90, 95]).tolist(),
            "diameter_percentiles_percent": np.percentile(relative * 100, [2.5, 5, 50, 95, 97.5]).tolist(),
            "reference_metrics_match": True, "reconstruction_glassplate_to_volume_um": 20000}


def fig08(output):
    truth = table(ROOT / "data/labels/validation/seed260745/collision_events.parquet")
    reference_curve = table(REFERENCE / "projected_xy_pr_curves.parquet")
    reference_summary = table(REFERENCE / "projected_xy_method_summary.csv")
    reference_target = table(REFERENCE / "projected_xy_target_recall.csv")
    reference_composition = table(REFERENCE / "projected_xy_fp_composition.csv")
    curves, summaries, target_rows, composition = [], [], [], []
    for dimension, method in [("3d", "ours"), ("2d", "depth_free"), ("minip2d", "minip_2d")]:
        pred = table(ROOT / f"data/predictions/{dimension}/validation/seed260745.parquet")
        pred["score"] = pred["relation_attention_3d" if dimension == "3d" else "relation_attention_2d"]
        curve, summary = event_curve({260745: pred}, {260745: truth})
        check(summary["event_ap"], reference_summary.loc[reference_summary.method.eq(method), "event_ap"].iloc[0],
              f"Figure 8 AP: {method}")
        curve["false_positive_events"] = curve.predicted_events - curve.matched_events
        curve["cohort"], curve["method"] = "terminal", method
        original = reference_curve.loc[reference_curve.method.eq(method)].reset_index(drop=True)
        check(curve[["predicted_events", "matched_events", "recall", "precision"]].to_numpy(),
              original[["predicted_events", "matched_events", "recall", "precision"]].to_numpy(), method)
        curves.append(curve)
        summaries.append({"cohort": "terminal", "method": method, **summary})
        for target in reference_target.loc[reference_target.method.eq(method), "target_recall"]:
            selected = curve.loc[curve.recall >= target].iloc[0]
            target_rows.append({"cohort": "terminal", "method": method, "target_recall": target,
                                **selected[["threshold", "recall", "precision", "false_positive_events", "predicted_events"]].to_dict()})
        point = curve.loc[curve.recall >= 0.8].iloc[0]
        selected = pred.loc[pred.score >= point.threshold].reset_index(drop=True).copy()
        selected["z_um"] = 0.0
        projected_truth = truth.copy()
        projected_truth["z_um"] = 0.0
        adjacency = candidate_truth_adjacency(selected, projected_truth, frame_tolerance=5, xy_tolerance_um=350,
                                               z_tolerance_um=15000)
        has_truth = sum(bool(neighbors) for neighbors in adjacency.values())
        counts = {"duplicate_or_competition": has_truth - int(point.matched_events),
                  "no_frame_xy_truth": len(selected) - has_truth}
        for category, count in counts.items():
            composition.append({"cohort": "terminal", "method": method, "category": category, "count": count})
    curves, summaries = pd.concat(curves, ignore_index=True), pd.DataFrame(summaries)
    targets, composition = pd.DataFrame(target_rows), pd.DataFrame(composition)
    check(targets.false_positive_events, reference_target.false_positive_events, "Figure 8(c) FP counts")
    keys = ["method", "category"]
    check(composition.set_index(keys).sort_index()["count"],
          reference_composition.set_index(keys).sort_index()["count"], "Figure 8(d) FP composition")
    for name, values in [("event_pr", curves), ("event_summary", summaries), ("target_recall", targets), ("fp_composition", composition)]:
        values.to_csv(output / f"{name}.csv", index=False)
    plot_311(8, output)
    return {"event_ap": dict(zip(summaries.method, summaries.event_ap, strict=True)),
            "fp_at_recall_0p8": {method: int(group["count"].sum()) for method, group in composition.groupby("method")},
            "all_curve_points_and_fp_counts_match": True}


def reviews():
    from figure_support.review_rank import validate_top_score_cohort
    labels = table(DATA / "review/visual_review_labels.csv")
    assert labels.status.value_counts().to_dict() == {"non_collision": 896, "collision": 185, "ambiguous": 39}
    assert labels.scene_num.nunique() == 14 and len(labels) == 1120
    assert len(validate_top_score_cohort(labels)) == len(labels)
    return labels


def fig11(output):
    from figure_support import review_scores
    labels = reviews()
    histogram = review_scores.plot(labels, output)
    histogram.to_csv(output / "score_histograms.csv", index=False)
    return {"binary_reviews": 1081, "confirmed": 185, "rejected": 896, "ambiguous_excluded": 39,
            "score_range": [0.98, 1.0], "panels": 1}


def fig12(output):
    from figure_support import review_rank, review_budget
    labels = reviews()
    all_review_rows = len(labels)
    curve = review_budget.validate_curve(review_rank.aggregate_by_rank(review_rank.validate_top_score_cohort(labels)))
    reference = table(REFERENCE / "review_topn_yield_precision_260715.csv")
    columns = ["top_n_per_scene", "candidates_reviewed", "cumulative_confirmed_collisions", "top_n_precision"]
    check(curve[columns], reference[columns], "Figure 12 review ranks and yields")
    labels = review_budget.validate_labels(labels, 185)
    operating, _, _ = review_budget.build_operating_table(curve, labels)
    curve.to_csv(output / "review_curve.csv", index=False)
    operating.to_csv(output / "review_operating_points.csv", index=False)
    plot_311(12, output)
    row = curve.loc[curve.top_n_per_scene.eq(30)].iloc[0]
    return {"all_review_rows": all_review_rows, "top_score_review_rows": len(labels),
            "confirmed_at_top30": int(row.cumulative_confirmed_collisions),
            "reviews_at_top30": int(row.candidates_reviewed), "coverage_at_top30": float(row.cumulative_confirmed_collisions / 185),
            "precision_at_top30": float(row.top_n_precision), "reference_curve_matches": True}


def fig13(output):
    from figure_support import physical_plot
    labels = reviews()
    events = table(DATA / "fig13/window_measurements.parquet")
    keys = ["scene_num", "vanished_track_id", "survivor_track_id", "candidate_frame_raw"]
    joined = events.merge(labels[keys + ["status"]], on=keys, validate="one_to_one")
    assert len(joined) == len(events) == 1081 and joined.status.eq(joined.review_status).all()
    assert events.review_stratum.eq("top_score").all()
    d1, d2 = events.pre_diameter_vanished_um, events.pre_diameter_survivor_um
    events["expected_post_diameter_um"] = np.cbrt(d1**3 + d2**3)
    events["expected_post_vy_m_s"] = (d1**3 * events.pre_vy_vanished_m_s + d2**3 * events.pre_vy_survivor_m_s) / (d1**3 + d2**3)
    events.loc[~events.velocity_available, "expected_post_vy_m_s"] = np.nan
    original = table(REFERENCE / "physical_consistency_events.csv")
    fields = ["expected_post_diameter_um", "expected_post_vy_m_s"]
    check(events[fields], original[fields], "Figure 13 conservation predictions")
    diameter, velocity = events.loc[events.diameter_available], events.loc[events.velocity_available]
    physical_plot.FIGURE_ROOT = output
    physical_plot.combined_scatter_figure(diameter, velocity)
    events.to_csv(output / "physical_consistency_events.csv", index=False)
    return {"binary_reviews": len(events), "confirmed_diameter": int(diameter.classification_role.eq("TP").sum()),
            "confirmed_velocity": int(velocity.classification_role.eq("TP").sum()),
            "rejected_diameter": int(diameter.classification_role.eq("FP").sum()),
            "rejected_velocity": int(velocity.classification_role.eq("FP").sum()), "conservation_predictions_match": True}


def gravity_analysis(convention="saved-figure"):
    """Compute either manuscript counts or the explicitly requested saved figures.

    poisson-only isolates the uncertainty change while keeping saved exposure.
    All comparisons retain the independent, unchanged paper reference tables.
    """
    from figure_support import gravity as g
    from figure_support.scene_index import SCENE_NUMBERS, paper_scene_index, paper_scene_label
    if convention not in {"manuscript", "saved-figure", "poisson-only"}:
        raise ValueError(f"Unknown gravity convention: {convention}")
    labels = reviews()
    fine = table(DATA / "gravity/fine_density_by_scene_5um.csv")
    counts = table(DATA / "gravity/recorded_particle_counts_by_scene_5um.csv")
    full_recording = convention == "manuscript"
    bootstrap_enabled = convention == "saved-figure"
    exposure = g.MANUSCRIPT_EXPOSURE_M3_S if full_recording else g.EXPOSURE_VOLUME_M3_S
    blocks = table(DATA / "gravity/time_block_density_by_scene_5um.csv") if bootstrap_enabled else None
    centers = np.arange(27.5, 300, 5)
    edges = np.arange(25, 301, 25)
    kernel = g.HallCorrectedLaminarKernel(ROOT / "src/ra3d/config/physics").kernel_m3_s(centers[:, None], centers[None, :])
    reference_kernel = table(DATA / "gravity/kernel_5um_grid.csv").pivot(index="diameter_1_um", columns="diameter_2_um", values="kernel_m3_s")
    check(kernel, reference_kernel.to_numpy(), "Gravity kernel", rtol=1e-12, atol=0)
    rng = np.random.default_rng(g.RANDOM_SEED) if bootstrap_enabled else None
    pooled = np.zeros((11, 11))
    pooled_boot = np.zeros((g.BOOTSTRAP_REPLICATES, 11, 11)) if bootstrap_enabled else None
    summaries = []
    for scene in SCENE_NUMBERS:
        original_density = fine.loc[fine.raw_scene_num.eq(scene)].sort_values("bin_low_um").number_density_cm3.to_numpy()
        source = counts.loc[counts.raw_scene_num.eq(scene)].sort_values("bin_low_um")
        assert len(source) == 55 and source.recorded_frames.eq(g.RECORDED_FRAME_COUNT).all()
        assert (source.first_10904_frame_count + source.last_13_frame_count).eq(source.recorded_particle_count).all()
        check(source.first_10904_frame_count / (g.EFFECTIVE_FRAME_COUNT * g.VOLUME_CM3),
              original_density, "Independent original particle-count check", rtol=1e-13, atol=1e-14)
        density = (source.recorded_particle_count.to_numpy() / (g.RECORDED_FRAME_COUNT * g.MANUSCRIPT_VOLUME_CM3)
                   if full_recording else original_density)
        expected = g.pair_prediction(density * 1e6, kernel, centers, edges, exposure_volume_m3_s=exposure)[0]
        pooled += np.nan_to_num(expected)
        mean = float(np.nansum(expected))
        sd = 0.0
        if bootstrap_enabled:
            block = blocks.loc[blocks.raw_scene_num.eq(scene)].pivot(index="block_index", columns="bin_low_um", values="number_density_cm3").to_numpy()
            sampled = rng.integers(0, g.TIME_BLOCKS, size=(g.BOOTSTRAP_REPLICATES, g.TIME_BLOCKS))
            bootstrap = g.pair_prediction(block[sampled].mean(axis=1) * 1e6, kernel, centers, edges,
                                          exposure_volume_m3_s=exposure)
            pooled_boot += np.nan_to_num(bootstrap)
            sd = float(np.std(np.nansum(bootstrap, axis=(1, 2)), ddof=1))
        observed = int((labels.scene_num.eq(scene) & labels.status.eq("collision")).sum())
        summaries.append({"row_type": "scene", "scene_index": paper_scene_index(scene), "raw_scene_num": scene,
            "display_label": paper_scene_label(scene), "confirmed_collision_lower_bound": observed,
            "predicted_expected_count": mean, "predicted_density_bootstrap_sd": sd,
            "predicted_total_sd": math.sqrt(mean + sd**2), "exposure_volume_m3_s": exposure})
    mean = float(pooled.sum())
    sd = float(pooled_boot.sum(axis=(1, 2)).std(ddof=1)) if bootstrap_enabled else 0.0
    summaries.append({"row_type": "total", "scene_index": None, "raw_scene_num": None, "display_label": "All scenes",
        "confirmed_collision_lower_bound": 185, "predicted_expected_count": mean, "predicted_density_bootstrap_sd": sd,
        "predicted_total_sd": math.sqrt(mean + sd**2), "exposure_volume_m3_s": len(SCENE_NUMBERS) * exposure})
    scenes = pd.DataFrame(summaries)
    scenes["predicted_poisson_sd"] = np.sqrt(scenes.predicted_expected_count)
    for count, rate in [("confirmed_collision_lower_bound", "confirmed_collision_rate_lower_bound_m3_s"),
                        ("predicted_expected_count", "predicted_collision_rate_m3_s"),
                        ("predicted_total_sd", "predicted_collision_rate_total_sd_m3_s")]:
        scenes[rate] = scenes[count] / scenes.exposure_volume_m3_s
    if bootstrap_enabled:
        density_sd, poisson_sd, total_sd = g.predictive_standard_deviation(pooled, pooled_boot)
    else:
        density_sd = np.zeros_like(pooled)
        poisson_sd = total_sd = np.sqrt(pooled)
    observed, _ = g.observed_collision_counts(labels, edges)
    original_pairs = table(REFERENCE / "collision_count_map_observed_vs_hall_25um.csv")
    fields = ["confirmed_collision_lower_bound", "predicted_expected_count", "predicted_density_bootstrap_sd", "predicted_total_sd"]
    pair_rows = []
    for row in original_pairs.itertuples():
        i, j = row.small_bin_index, row.large_bin_index
        actual = [observed[i, j], pooled[i, j], density_sd[i, j], total_sd[i, j]]
        pair_rows.append({"small_bin_index": i, "large_bin_index": j,
                          "predicted_poisson_sd": float(poisson_sd[i, j]), **dict(zip(fields, actual, strict=True))})
    pairs = pd.DataFrame(pair_rows)
    original_scenes = table(REFERENCE / "scene_collision_counts_observed_vs_hall_260715.csv")
    if bootstrap_enabled:
        check(scenes[fields], original_scenes[fields], "Figure 14 saved counts and uncertainties")
        check(pairs[fields], original_pairs[fields], "Figure 15 saved pair cells")
    check(scenes.confirmed_collision_lower_bound, original_scenes.confirmed_collision_lower_bound, "Figure 14 confirmed counts")
    check(pairs.confirmed_collision_lower_bound, original_pairs.confirmed_collision_lower_bound, "Figure 15 confirmed counts")
    return g, scenes, pairs, observed, pooled, total_sd, edges


def gravity_comparison(scenes, pairs):
    """Report differences from the committed figure data without replacing them."""
    original_scenes = table(REFERENCE / "scene_collision_counts_observed_vs_hall_260715.csv")
    original_pairs = table(REFERENCE / "collision_count_map_observed_vs_hall_25um.csv")
    columns = ["confirmed_collision_lower_bound", "predicted_expected_count", "predicted_poisson_sd",
               "predicted_density_bootstrap_sd", "predicted_total_sd", "exposure_volume_m3_s",
               "confirmed_collision_rate_lower_bound_m3_s", "predicted_collision_rate_m3_s"]
    summary = {}
    for name, current, original in [("scenes", scenes, original_scenes), ("diameter_pairs", pairs, original_pairs)]:
        summary[name] = {}
        for column in columns:
            if column not in current or column not in original:
                continue
            a, b = current[column].to_numpy(float), original[column].to_numpy(float)
            equal = np.isclose(a, b, rtol=1e-10, atol=1e-9, equal_nan=True)
            summary[name][column] = {"match": bool(equal.all()), "different_rows": int((~equal).sum()),
                                     "max_absolute_difference": float(np.max(np.abs(a - b)))}
    summary["all_reference_values_match"] = all(v["match"] for group in summary.values() for v in group.values())
    return summary


def main():
    global PLOT_PYTHON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", default=["all"], choices=["all"] + list(map(str, FIGURES)))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/paper-figures")
    parser.add_argument("--verify-rendered", action="store_true", help="Require exact panel-pixel agreement with the paper figures using Poppler")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--gravity-convention", choices=["saved-figure", "manuscript", "poisson-only"], default="saved-figure",
                        help="Figures 14–15: saved-figure reproduces the paper plots (default); manuscript and poisson-only are alternative calculations")
    parser.add_argument("--plot-python", type=Path, default=PLOT_PYTHON,
                        help="Python in the separate Matplotlib 3.11.1 plot environment")
    args = parser.parse_args()
    from figure_support.delivery import assemble_and_verify, check_renderer
    check_renderer(args.pdftoppm)
    chosen = FIGURES if "all" in args.figures else sorted(set(map(int, args.figures)))
    if mpl.__version__ != "3.10.9":
        parser.error("Run with the Matplotlib 3.10.9 core environment; see requirements-figures-reference.txt")
    # Keep the venv interpreter path: resolving its symlink bypasses that venv.
    PLOT_PYTHON = args.plot_python.absolute()
    if set(chosen) & {7, 8, 12}:
        try:
            version = subprocess.check_output([str(PLOT_PYTHON), "-c", "import matplotlib; print(matplotlib.__version__)"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            parser.error("Create .venv-figures311 with requirements-figures311-reference.txt, or pass --plot-python")
        if version != "3.11.1":
            parser.error("--plot-python must have Matplotlib 3.11.1")
    output = args.output.resolve()
    for name in ["data", "src", "scripts", "reference", "provenance", "vendor"]:
        if output.is_relative_to(ROOT / name):
            parser.error("Choose an output outside the immutable release inputs")
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new, empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    register_fonts()
    paper_reference = json.loads((ROOT / "provenance/paper_figures.json").read_text())
    report = {"status": "IN_PROGRESS", "paper_commit": paper_reference["paper_commit"], "figures": {}}
    gravity_values = None
    for number in chosen:
        directory = output / f"figure{number:02d}"
        directory.mkdir()
        print(f"Reproducing Figure {number}...", flush=True)
        mpl.rcdefaults()
        if number in [14, 15]:
            if gravity_values is None:
                gravity_values = gravity_analysis(args.gravity_convention)
            g, scenes, pairs, observed, expected, sd, edges = gravity_values
            if number == 14:
                g.save_scene_comparison(directory / "scene_collision_counts_observed_vs_hall_260715",
                                       scenes.loc[scenes.row_type.eq("scene")], scenes.iloc[-1])
                scenes.to_csv(directory / "scene_counts.csv", index=False)
            else:
                g.save_pair_map(directory / "collision_count_map_observed_vs_hall_260715", observed, expected, sd, edges)
                pairs.to_csv(directory / "diameter_pair_counts.csv", index=False)
            comparison = gravity_comparison(scenes, pairs)
            write_json(directory / "paper_reference_comparison.json", comparison)
            result = {"gravity_convention": args.gravity_convention, "confirmed": int(observed.sum()), "predicted_total": float(expected.sum()),
                      "pooled_total_sd": float(scenes.iloc[-1].predicted_total_sd),
                      "pooled_observed_rate_m3_s": float(scenes.iloc[-1].confirmed_collision_rate_lower_bound_m3_s),
                      "pooled_exposure_volume_m3_s": float(scenes.iloc[-1].exposure_volume_m3_s),
                      "reference_tables_match": comparison["all_reference_values_match"],
                      "bootstrap_replicates": 4000 if args.gravity_convention == "saved-figure" else 0,
                      "bootstrap_seed": 260724 if args.gravity_convention == "saved-figure" else None,
                      "diameter_assignment": "hard_25_um_bins"}
        else:
            result = globals()[f"fig{number:02d}"](directory)
        report["figures"][str(number)] = result
        write_json(output / "numerical_verification.json", report)
    report["status"] = "PASS"
    write_json(output / "numerical_verification.json", report)
    report["rendering"] = assemble_and_verify(output, chosen, verify=args.verify_rendered, pdftoppm=args.pdftoppm)
    if not report["rendering"]["all_reference_pixels_match"]:
        report["status"] = "COMPUTED_WITH_PAPER_DIFFERENCES"
    write_json(output / "verification.json", report)
    print(json.dumps({"status": report["status"], "figures": chosen, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
