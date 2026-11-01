"""Recompute every selected image feature at three iterations, on fixed inputs."""
from common import *
import argparse
import time
import torch
import yaml
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d.features import image_features
from ra3d.config import KEY_COLUMNS, IMAGE_FEATURES
from ra3d._core.features import add_derived_features


def run(seed, device):
    marker = data_path(seed, 'images3.json')
    if marker.exists():
        emit(status='ALREADY_COMPLETE', seed=seed)
        return
    started = time.monotonic()
    config = yaml.safe_load((ARCHIVE / f'config/seed{seed}-collision.yaml').read_text())
    config['optics']['phase_iterations'] = 3
    path = ROOT / f'config/seed{seed}-collision3.yaml'
    path.write_text(yaml.safe_dump(config))
    acquisition = Acquisition.load(path)
    assert acquisition.reconstruction_start_um == 74900
    assert acquisition.padding_px == 2048
    requested = read_table(feature_path(seed, 'image_requested.parquet'))
    tracks = read_table(tracks_path(seed, 'geometry_smoothed.parquet'))
    torch.set_num_threads(2)
    recon = Reconstructor(acquisition, device)
    image = image_features(requested, tracks, recon, progress=lambda v: emit(seed=seed, **v),
                           checkpoint_every=250, checkpoint_path=data_path(seed, 'images3-checkpoint.npz'))
    # Retain the exact original union, static scores, missingness and row order.
    original = read_table(feature_path(seed, 'paper92.parquet'))
    updated = original.drop(columns=IMAGE_FEATURES).merge(image[KEY_COLUMNS + IMAGE_FEATURES],
                        on=KEY_COLUMNS, how='left', sort=False, validate='one_to_one')
    updated = add_derived_features(updated)
    pd.testing.assert_frame_equal(original.drop(columns=IMAGE_FEATURES),
                                 updated[original.drop(columns=IMAGE_FEATURES).columns], check_exact=True)
    # Unselected candidates retain missing raw features and zero-valued
    # derived indicators. A measured component area may become unavailable
    # when the phase intervention changes whether a common blob is found.
    for col in ['rendered_pre_frames','skipped_pre_frames']:
        np.testing.assert_allclose(original[col],updated[col],equal_nan=True,rtol=0,atol=0)
    for col in IMAGE_FEATURES:
        if col != 'same_component_area_median_px':
            assert np.array_equal(original[col].isna(),updated[col].isna()),col
    write_table(image[KEY_COLUMNS + IMAGE_FEATURES], data_path(seed, 'image_features3.parquet'))
    write_table(updated, data_path(seed, 'paper92_phase3.parquet'))
    differences = []
    for col in IMAGE_FEATURES:
        a = original[col].to_numpy(float); b = updated[col].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        differences.append({'feature': col, 'valid': int(finite.sum()),
                            'changed': int(np.count_nonzero(a[finite] != b[finite])),
                            'missing_phase12':int(np.isnan(a).sum()), 'missing_phase3':int(np.isnan(b).sum()),
                            'mean_absolute_difference': float(np.abs(a[finite]-b[finite]).mean())})
    write_table(pd.DataFrame(differences), data_path(seed, 'feature_differences.csv'))
    write_json({'status': 'PASS', 'seed': seed, 'iterations': 3, 'padding_px': 2048,
                'front_um': 74900, 'image_candidates': len(image), 'full_candidates': len(updated),
                'seconds': time.monotonic()-started, 'candidate_membership_and_static_features_identical': True,
                'missingness_identical': bool(np.array_equal(original[IMAGE_FEATURES].isna(),updated[IMAGE_FEATURES].isna())),
                'acquisition_selection_and_rendered_frames_identical':True,
                'source_full_sha256': sha256(feature_path(seed, 'paper92.parquet')),
                'source_tracks_sha256': sha256(tracks_path(seed, 'geometry_smoothed.parquet')),
                'output_sha256': sha256(data_path(seed, 'paper92_phase3.parquet'))}, marker)
    data_path(seed, 'images3-checkpoint.npz').unlink(missing_ok=True)
    emit(status='PASS', seed=seed, seconds=time.monotonic()-started)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--device', required=True)
    args = p.parse_args()
    for seed in args.seeds:
        run(seed, args.device)
