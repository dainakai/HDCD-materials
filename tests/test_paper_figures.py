"""Scientific safeguards for review denominators and unordered diameter-pair counts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figure_support import gravity, review_budget, review_rank  # noqa: E402


def labels():
    return pd.read_csv(ROOT / "data/experimental/reviews.csv", float_precision="round_trip")


def test_ambiguous_reviews_count_toward_budget_but_not_binary_outcomes():
    reviewed = labels()
    assert reviewed.status.value_counts().to_dict() == {"non_collision": 922, "collision": 186, "ambiguous": 12}
    assert reviewed.loc[reviewed.status.eq("ambiguous"), "binary_label"].isna().all()
    cohort = review_rank.validate_top_score_cohort(reviewed)
    curve = review_budget.validate_curve(review_rank.aggregate_by_rank(cohort))
    indexed = curve.set_index('top_n_per_scene')
    assert len(cohort) == 1120
    assert reviewed.groupby('scene_num').size().eq(80).all()
    assert indexed.loc[24, 'candidates_reviewed'] == 336
    assert indexed.loc[24, 'cumulative_confirmed_collisions'] == 173
    assert indexed.loc[24, 'top_n_precision'] == 173 / 336
    assert indexed.loc[25, 'top_n_precision'] < .5
    assert indexed.loc[34, 'cumulative_confirmed_collisions'] == 177
    assert indexed.loc[34, 'top_n_precision'] == 177 / 476


def test_physical_panels_use_only_the_current_binary_review_cohort():
    reviewed = labels()
    events = pd.read_parquet(ROOT / 'data/experimental/event_measurements.parquet')
    assert len(events) == len(reviewed) == 1120
    assert events.status.value_counts().to_dict() == reviewed.status.value_counts().to_dict()
    assert events.loc[events.diameter_available, 'classification_role'].value_counts().to_dict() == {'FP': 908, 'TP': 184, 'ambiguous': 12}
    assert events.loc[events.velocity_available & events.classification_role.isin(['TP', 'FP']), 'classification_role'].value_counts().to_dict() == {'FP': 822, 'TP': 179}


def test_equal_coarse_diameter_bin_retains_unequal_fine_diameter_collisions():
    # Two unequal droplets share the 25–50 µm display bin. Their mutual kernel
    # is nonzero even though an exactly equal-diameter gravitational kernel is zero.
    centers = np.array([27.5, 32.5])
    density = np.array([2.0, 3.0])
    kernel = np.array([[0.0, 7.0], [7.0, 0.0]])
    counts = gravity.pair_prediction(density, kernel, centers, np.array([25.0, 50.0]))
    expected = gravity.MANUSCRIPT_EXPOSURE_M3_S * 2 * 3 * 7
    np.testing.assert_allclose(counts[0, 0, 0], expected, rtol=1e-15)


def test_gravity_uses_full_volume_unordered_pairs_and_poisson_only(tmp_path):
    from reproduce_experimental import calculate
    result = calculate(tmp_path)
    scenes = pd.read_csv(tmp_path / 'scene_collision_counts.csv')
    total = scenes.iloc[-1]
    volume = .01024 * .01024 * .1023
    exposure = 14 * 10917 / 4000 * volume
    np.testing.assert_allclose(total.exposure_volume_m3_s, exposure, rtol=1e-12)
    np.testing.assert_allclose(total.confirmed_collision_rate_lower_bound_m3_s, 186 / exposure, rtol=1e-12)
    np.testing.assert_allclose(scenes.predicted_total_sd**2, scenes.predicted_expected_count, rtol=1e-12)
    assert scenes.predicted_density_bootstrap_sd.eq(0).all()
    assert result['diameter_pairs'] == 185
    density = pd.read_csv(ROOT / 'data/experimental/diameter_density_by_scene.csv')
    counts = density.pivot(index='scene_num', columns='diameter_low_um', values='recorded_particle_count').to_numpy()
    dn = counts / (10917 * volume)
    centers = np.arange(27.5, 300, 5)
    kernel = gravity.HallCorrectedLaminarKernel(ROOT / 'src/ra3d/config/physics').kernel_m3_s(centers[:, None], centers[None, :])
    expected = .5 * (10917 / 4000 * volume) * np.einsum('si,ij,sj->', dn, kernel, dn)
    np.testing.assert_allclose(total.predicted_expected_count, expected, rtol=1e-12)
