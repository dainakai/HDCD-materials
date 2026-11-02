"""Validate paired raw frames/calibration; write version-bound acquisition contracts."""
from common import *
import csv
import io
import shutil
import struct
import numpy as np
import yaml
from PIL import Image


def main():
    freeze = verify_engine()
    result = []
    variables = yaml.safe_load((RAW / 'variables.yaml').read_text())
    assert [variables[x] for x in ['datlen', 'dx', 'dz', 'slices', 'lambda']] == [1024, 10., 100., 1024, .6328]
    for n in SCENES:
        historical = read(PREPARATION / f'scenes/scene{n:04d}.json')
        root = scene(n)
        (root / 'input').mkdir(parents=True, exist_ok=True)
        prepared = root / 'prepared'
        calib = Path(historical['calibration_source'])
        coefficients = np.loadtxt(calib / 'coeffs.txt')
        coefficients = coefficients[:, 0] if coefficients.ndim == 2 else coefficients
        focal = np.loadtxt(calib / 'focal_adjustment.txt')
        focal = focal[:, 0] if focal.ndim == 2 else focal
        assert coefficients.shape == (12,) and np.isfinite(coefficients).all()
        assert len(focal) >= 2 and np.isfinite(focal[:2]).all()
        assert abs(focal[0] + 20000. - historical['historical_front_um']) < 1e-6
        offset = 74900. - float(focal[0])
        cameras = []
        for camera in ['C001', 'C002']:
            directory = RAW / f'00_img/{camera}H001S{n:04d}'
            paths = sorted(directory.glob('*.bmp'))
            indices = [int(p.stem[-6:]) for p in paths]
            assert indices == list(range(1, FRAMES + 1)), (n, camera, len(indices))
            rows = [{'raw_index': i, 'name': p.name, 'size_bytes': p.stat().st_size,
                     'mtime_ns': p.stat().st_mtime_ns} for i, p in zip(indices, paths)]
            assert min(x['size_bytes'] for x in rows) > 1024 * 1024
            file = root / f'input/{camera}-raw-manifest.csv'
            with file.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            samples = []
            for index in [0, FRAMES // 2, FRAMES - 1]:
                p = paths[index]
                with Image.open(p) as image:
                    image.load()
                    assert image.size == (1024, 1024) and image.mode == 'L', (p, image.size, image.mode)
                samples.append({'raw_index': index + 1, 'sha256': sha(p)})
            cameras.append({'camera': camera, 'directory': str(directory), 'frames': len(paths),
                            'raw_indices': [1, FRAMES], 'total_bytes': sum(x['size_bytes'] for x in rows),
                            'file_manifest_sha256': sha(file), 'sample_content_hashes': samples})
        destination = prepared / '02_calibration' / calib.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['coeffs.txt', 'focal_adjustment.txt']:
            shutil.copy2(calib / name, destination / name)
        prepared_variables = dict(variables)
        prepared_variables.update(expdate=260715, priter=3)
        (prepared / 'variables.yaml').write_text(yaml.safe_dump(prepared_variables, sort_keys=False))
        contract = {'status': 'PASS', 'scene': n, 'run_id': RUN_ID, 'frames': FRAMES,
                    'cameras': cameras, 'primary': 'C002', 'secondary': 'C001',
                    'pipeline_frame_to_raw_index': 1, 'calibration_scene': calib.name,
                    'calibration_hashes': {name: sha(calib / name) for name in ['coeffs.txt', 'focal_adjustment.txt']},
                    'calibrated_camera_to_glassplate_um': float(focal[0]),
                    'phase_distance_um': float(focal[1]), 'glassplate_to_volume_um': offset,
                    'historical_front_um': historical['historical_front_um'],
                    'reconstruction_start_um': 74900, 'reconstruction_last_um': 177200,
                    'phase_iterations': 3, 'measurement_padding_px': 1536, 'collision_padding_px': 2048,
                    'minip_plane_indices': list(range(0, 1024, 2)) + [1023],
                    'review_budget': 80, 'model_json_sha256': sha(MODEL / 'model.json'),
                    'validation_scope': 'All filename indices and file sizes; full decode and SHA256 for first/middle/last frame per camera. All frames decoded during background processing.',
                    'checked_at': stamp()}
        target = root / 'input/contract.json'
        if target.exists():
            old = read(target)
            assert {k:v for k,v in old.items() if k != 'checked_at'} == {k:v for k,v in contract.items() if k != 'checked_at'}
        else:
            save(target, contract)
        frame_file = root / 'input/frames.csv'
        with frame_file.open('w', newline='') as stream:
            writer = csv.writer(stream);writer.writerow(['frame','primary','secondary'])
            for f in range(FRAMES):
                writer.writerow([f, str(prepared / f'01_bkgrem/C002H001S{n:04d}/{f+1:06d}.png'),
                                 str(prepared / f'01_bkgrem/C001H001S{n:04d}/{f+1:06d}.png')])
        acquisition = {'schema_version': 1, 'frames': 'frames.csv',
                       'units': {'image_size_px': 1024, 'pixel_pitch_um': 10., 'slice_spacing_um': 100., 'fps': 4000.},
                       'optics': {'wavelength_um': .6328, 'phase_distance_um': float(focal[1]),
                                  'reconstruction_start_um': 74900., 'padding_px': 2048, 'phase_iterations': 3,
                                  'calibration_coefficients': str(destination / 'coeffs.txt'), 'mode': 'dual_phase_retrieval'}}
        (root / 'input/acquisition.yaml').write_text(yaml.safe_dump(acquisition, sort_keys=False))
        result.append({'scene': n, 'frames_per_camera': FRAMES, 'calibration': calib.name,
                       'phase_distance_um': float(focal[1]), 'front_um': 74900, 'glassplate_offset_um': offset,
                       'input_contract_sha256': sha(target)})
        emit(stage='inputs_verified', **result[-1])
    save(ROOT / 'input-inventory.json', {'status':'PASS', 'checked_at':stamp(), 'disk_uuid':'54CEE478CEE453B0',
          'source_mount_mode':'read-only', 'scenes':result, 'paired_frames':FRAMES*len(SCENES),
          'raw_images':FRAMES*len(SCENES)*2, 'full_source_content_hashing':False})
    contract = read(PREPARATION / 'review_contract.json')
    contract.update(run_id=RUN_ID, status='INPUTS_VALIDATED_PROCESSING_PENDING',
                    budget_confirmation='User explicitly confirmed top 80 per scene on 2026-11-02',
                    model_json_sha256=sha(MODEL / 'model.json'))
    save(ROOT / 'review-contract.json', contract)
    emit(status='PASS', scenes=len(result), paired_frames=FRAMES*len(SCENES))


if __name__ == '__main__':
    main()
