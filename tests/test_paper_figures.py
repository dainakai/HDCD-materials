"""Scientific safeguards for review denominators and unordered diameter-pair counts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figure_support import gravity, review_budget, review_rank  # noqa: E402


def labels():
    return pd.read_csv(ROOT / "data/paper_figures/review/visual_review_labels.csv", float_precision="round_trip")


def test_ambiguous_reviews_count_toward_budget_but_not_binary_outcomes():
    reviewed = labels()
    assert reviewed.status.value_counts().to_dict() == {"non_collision": 896, "collision": 185, "ambiguous": 39}
    assert reviewed.loc[reviewed.status.eq("ambiguous"), "binary_label"].isna().all()
    cohort = review_rank.validate_top_score_cohort(reviewed)
    curve = review_budget.validate_curve(review_rank.aggregate_by_rank(cohort))
    at30 = curve.set_index("top_n_per_scene").loc[30]
    assert len(cohort) == 1120
    assert len(reviewed) == len(cohort)
    assert reviewed.review_stratum.eq("top_score").all()
    assert reviewed.groupby("scene_num").size().eq(80).all()
    assert at30.candidates_reviewed == 420
    assert at30.cumulative_confirmed_collisions == 177
    assert at30.cumulative_ambiguous == 18
    assert at30.top_n_precision == 177 / 420
    # Treating the 18 ambiguous reviews as absent would falsely improve precision.
    assert at30.top_n_precision < 177 / (420 - 18)


def test_physical_panels_use_only_the_current_binary_review_cohort():
    reviewed = labels()
    binary = reviewed.loc[reviewed.status.ne("ambiguous")]
    events = pd.read_parquet(ROOT / "data/paper_figures/fig13/window_measurements.parquet")
    keys = ["scene_num", "vanished_track_id", "survivor_track_id", "candidate_frame_raw"]
    joined = events.merge(binary[keys + ["status"]], on=keys, validate="one_to_one")
    assert len(joined) == len(events) == len(binary) == 1081
    assert joined.review_status.eq(joined.status).all()
    assert events.loc[events.diameter_available, "classification_role"].value_counts().to_dict() == {"FP": 877, "TP": 185}
    assert events.loc[events.velocity_available, "classification_role"].value_counts().to_dict() == {"FP": 803, "TP": 180}


def test_equal_coarse_diameter_bin_retains_unequal_fine_diameter_collisions():
    # Two unequal droplets share the 25–50 µm display bin. Their mutual kernel
    # is nonzero even though an exactly equal-diameter gravitational kernel is zero.
    centers = np.array([27.5, 32.5])
    density = np.array([2.0, 3.0])
    kernel = np.array([[0.0, 7.0], [7.0, 0.0]])
    counts = gravity.pair_prediction(density, kernel, centers, np.array([25.0, 50.0]))
    expected = gravity.MANUSCRIPT_EXPOSURE_M3_S * 2 * 3 * 7
    np.testing.assert_allclose(counts[0, 0, 0], expected, rtol=1e-15)


def test_manuscript_gravity_uses_full_counts_volume_and_poisson_only():
    from reproduce_figures import gravity_analysis
    _, scenes, pairs, _, expected, sd, _ = gravity_analysis("manuscript")
    total = scenes.iloc[-1]
    volume_m3 = 0.01024 * 0.01024 * 0.1023
    exposure = 14 * 10917 / 4000 * volume_m3
    np.testing.assert_allclose(total.exposure_volume_m3_s, exposure, rtol=1e-15)
    np.testing.assert_allclose(total.confirmed_collision_rate_lower_bound_m3_s, 185 / exposure, rtol=1e-15)
    np.testing.assert_allclose(scenes.predicted_total_sd**2, scenes.predicted_expected_count, rtol=1e-15)
    np.testing.assert_allclose(sd**2, expected, rtol=1e-15)
    assert scenes.predicted_density_bootstrap_sd.eq(0).all()
    assert pairs.predicted_density_bootstrap_sd.eq(0).all()
    # Independent full-matrix sum: unordered pairs are half of all ordered pairs.
    counts = pd.read_csv(ROOT / "data/paper_figures/gravity/recorded_particle_counts_by_scene_5um.csv")
    density = counts.pivot(index="raw_scene_num", columns="bin_low_um", values="recorded_particle_count").to_numpy()
    density = density / (10917 * volume_m3)
    centers = np.arange(27.5, 300, 5)
    kernel = gravity.HallCorrectedLaminarKernel(ROOT / "src/ra3d/config/physics").kernel_m3_s(centers[:, None], centers[None, :])
    direct = 0.5 * (10917 / 4000 * volume_m3) * np.einsum("si,ij,sj->", density, kernel, density)
    np.testing.assert_allclose(total.predicted_expected_count, direct, rtol=1e-13)


def test_gravity_histogram_keeps_final_recorded_frames(tmp_path):
    from prepare_gravity_counts import count_particles
    source = tmp_path / "particles.csv"
    pd.DataFrame({"frame": [0, 10903, 10904, 10916, 10],
                  "final_diameter_um": [27.5, 32.5, 27.5, 300.0, np.nan]}).to_csv(source, index=False)
    early, tail, audit = count_particles(source)
    assert early.sum() == tail.sum() == 2
    assert early[0] == tail[0] == tail[-1] == 1
    assert audit["frame_max"] == 10916
