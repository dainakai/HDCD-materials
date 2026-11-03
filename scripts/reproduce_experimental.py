#!/usr/bin/env python3
"""Recompute review statistics, physical diagnostics and experimental figures."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import pandas as pd
import polars as pl
from figure_support import review_rank, gravity, population_size, population_count

FRAMES = 10917
FPS = 4000.
VOLUME_CM3 = 10.24 * 10.24 * 102.3 / 1000
EXPOSURE = VOLUME_CM3 * 1e-6 * FRAMES / FPS


def calculate(output):
    source = REPO / 'data/experimental'
    labels = pd.read_csv(source / 'reviews.csv', float_precision='round_trip')
    top = review_rank.validate_top_score_cohort(labels)
    curve = review_rank.aggregate_by_rank(top)
    confirmed = int(labels.status.eq('collision').sum())
    curve['coverage_of_reviewed_confirmed'] = curve.cumulative_confirmed_collisions / confirmed
    curve['binary_only_precision'] = curve.cumulative_confirmed_collisions / (curve.cumulative_confirmed_collisions + curve.cumulative_non_collisions)
    curve.to_csv(output / 'review_budget.csv', index=False)
    counts = pd.crosstab(labels.scene_num, labels.status).reindex(columns=review_rank.STATUS_ORDER).reset_index()
    counts.insert(0, 'scene_index', range(1, 15))
    counts['reviewed'] = 80
    counts.to_csv(output / 'review_counts_by_scene.csv', index=False)
    events = pd.read_parquet(source / 'event_measurements.parquet')
    dv, ds = events.pre_diameter_vanished_um, events.pre_diameter_survivor_um
    valid_parent = np.isfinite(dv) & np.isfinite(ds) & dv.gt(0) & ds.gt(0)
    expected_diameter = np.cbrt(dv**3 + ds**3).where(valid_parent)
    velocity_valid = (events[['pre_vy_vanished_n', 'pre_vy_survivor_n', 'post_vy_n']].min(axis=1).ge(3)
                      & np.isfinite(events[['pre_vy_vanished_m_s', 'pre_vy_survivor_m_s', 'post_vy_unfiltered_m_s']]).all(axis=1)
                      & valid_parent)
    expected_velocity = ((dv**3 * events.pre_vy_vanished_m_s + ds**3 * events.pre_vy_survivor_m_s) / (dv**3 + ds**3)).where(velocity_valid)
    np.testing.assert_allclose(expected_diameter, events.expected_post_diameter_um, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(expected_velocity, events.expected_post_vy_m_s, rtol=1e-12, equal_nan=True)
    events['expected_post_diameter_um'] = expected_diameter
    events['expected_post_vy_m_s'] = expected_velocity
    events['measured_post_diameter_um'] = events.post_diameter_um
    events['measured_post_vy_m_s'] = events.post_vy_unfiltered_m_s.where(velocity_valid)
    events['diameter_available'] = valid_parent & np.isfinite(events.post_diameter_um) & events.post_diameter_um.gt(0)
    events['velocity_available'] = velocity_valid
    events.to_parquet(output / 'physical_events.parquet', index=False)
    metrics = []
    for role in ['TP', 'FP']:
        for quantity, suffix, available in [('diameter_um', 'diameter_um', 'diameter_available'), ('velocity_m_s', 'vy_m_s', 'velocity_available')]:
            part = events.loc[events.classification_role.eq(role) & events[available]]
            error = part[f'measured_post_{suffix}'] - part[f'expected_post_{suffix}']
            metrics.append({'role': role, 'quantity': quantity, 'n': len(part), 'rmse': float(np.sqrt(np.mean(error**2))),
                            'bias': float(error.mean()), 'mae': float(error.abs().mean())})
    pd.DataFrame(metrics).to_csv(output / 'physical_error_summary.csv', index=False)
    density = pd.read_csv(source / 'diameter_density_by_scene.csv', float_precision='round_trip')
    centers = np.arange(27.5, 300, 5.)
    edges = np.arange(25., 325., 25.)
    kernel = gravity.HallCorrectedLaminarKernel(REPO / 'src/ra3d/config/physics')
    aa, bb = np.meshgrid(centers, centers, indexing='ij')
    matrix = kernel.kernel_m3_s(aa, bb)
    predictions, scenes = [], []
    for idx, (scene, part) in enumerate(density.groupby('scene_num', sort=True), 1):
        dn = part.recorded_particle_count.to_numpy() / (FRAMES * VOLUME_CM3)
        np.testing.assert_allclose(dn, part.number_density_cm3, rtol=1e-12)
        predicted = gravity.pair_prediction(dn * 1e6, matrix, centers, edges, exposure_volume_m3_s=EXPOSURE)[0]
        np.testing.assert_allclose(np.nansum(predicted), EXPOSURE * .5 * np.einsum('i,ij,j', dn * 1e6, matrix, dn * 1e6), rtol=1e-12)
        predictions.append(np.nan_to_num(predicted))
        scenes.append(dict(row_type='scene', scene_index=idx, raw_scene_num=scene, display_label=f'Scene {idx}',
                           confirmed_collision_lower_bound=int((labels.scene_num.eq(scene) & labels.status.eq('collision')).sum()),
                           predicted_expected_count=float(np.nansum(predicted)), exposure_volume_m3_s=EXPOSURE))
    pooled = np.sum(predictions, axis=0)
    pairs = labels.copy()
    invalid = ~np.isfinite(pairs[['pre_diameter_vanished_um', 'pre_diameter_survivor_um']]).all(axis=1) | pairs.pre_diameter_vanished_um.le(0) | pairs.pre_diameter_survivor_um.le(0)
    pairs.loc[invalid, ['pre_diameter_small_um', 'pre_diameter_large_um']] = np.nan
    observed, pair_audit = gravity.observed_collision_counts(pairs, edges)
    pair_table = pd.DataFrame([dict(small_bin_index=i, large_bin_index=j, small_low_um=edges[i], small_high_um=edges[i+1],
                                   large_low_um=edges[j], large_high_um=edges[j+1], confirmed_collision_lower_bound=int(observed[i, j]),
                                   predicted_expected_count=pooled[i, j], predicted_poisson_sd=np.sqrt(pooled[i, j]))
                              for i, j in zip(*np.triu_indices(11))])
    pair_table.to_csv(output / 'diameter_pair_counts.csv', index=False)
    scenes.append(dict(row_type='total', scene_index=None, raw_scene_num=None, display_label='All scenes',
                       confirmed_collision_lower_bound=confirmed, predicted_expected_count=float(pooled.sum()), exposure_volume_m3_s=14*EXPOSURE))
    scene_table = pd.DataFrame(scenes)
    scene_table['predicted_poisson_sd'] = np.sqrt(scene_table.predicted_expected_count)
    scene_table['predicted_density_bootstrap_sd'] = 0.
    scene_table['predicted_total_sd'] = scene_table.predicted_poisson_sd
    for count, rate in [('confirmed_collision_lower_bound', 'confirmed_collision_rate_lower_bound_m3_s'),
                        ('predicted_expected_count', 'predicted_collision_rate_m3_s'), ('predicted_total_sd', 'predicted_collision_rate_total_sd_m3_s')]:
        scene_table[rate] = scene_table[count] / scene_table.exposure_volume_m3_s
    scene_table.to_csv(output / 'scene_collision_counts.csv', index=False)
    frames = pl.read_parquet(source / 'frame_metrics.parquet')
    population_count.scene_summary(frames, sorted(frames['scene'].unique())).write_csv(output / 'scene_population_statistics.csv')
    bins, samples = population_size.load_scene_curves(source / 'diameter_density_by_scene.csv')
    low, high = population_size.bootstrap_mean_interval(samples, replicates=50000, seed=260715, chunk_size=1000)
    bins.assign(mean=samples.mean(axis=0), ci025=low, ci975=high).to_csv(output / 'diameter_mean_ci.csv', index=False)
    for name in ['review_budget', 'review_counts_by_scene', 'scene_collision_counts', 'diameter_pair_counts', 'diameter_mean_ci', 'scene_population_statistics']:
        actual = pd.read_csv(output / (name + '.csv'))
        reference = pd.read_csv(REPO / 'reference/experimental' / (name + '.csv'))
        for col in reference.select_dtypes('number'):
            np.testing.assert_allclose(actual[col], reference[col], rtol=1e-11, atol=1e-10, equal_nan=True)
    result = {'status': 'PASS', 'reviewed': len(labels), 'confirmed': confirmed, 'diameter_pairs': int(observed.sum()),
              'physical': metrics, 'expected_collisions': float(pooled.sum()), 'poisson_sd': float(np.sqrt(pooled.sum())),
              'coverage_denominator': confirmed, 'frames_per_scene': FRAMES, 'volume_cm3': VOLUME_CM3}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--figures', action='store_true')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose an empty output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(calculate(a.output), indent=2))
    if a.figures:
        from figure_support.experimental import main as plot
        plot(a.output)


if __name__ == '__main__':
    main()
