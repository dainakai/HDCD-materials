#!/usr/bin/env python3
"""Find well-localized collisions whose six complete crops are inside the hologram."""
import argparse
import json
import pickle
from collections import Counter

from search import REPO, ROOT, original
import numpy as np
import pandas as pd

OUT = REPO.parent / 'audit/revision-figure9-interior-20261103'
OUT.mkdir(exist_ok=True)
DELTAS = range(-4, 2)
PREVIEW_CASES = [
    ('N1', 3958, 12754, 12764, 'Large-particle option; disclose the 4.31px maximum center error.'),
    ('N2', 5894, 19387, 19524, 'Balanced parent sizes with maximum center error below 3px.'),
    ('N3', 163, 65, 325, 'Relatively large parents with approximately 2px maximum center error.'),
    ('N4', 1897, 4252, 4359, 'Another balanced-parent alternative.'),
    ('N5', 4178, 12913, 13140, 'Farther inside the hologram with relatively unobstructed target images.'),
    ('N6', 5472, 17670, 17995, 'Smaller particles far inside the hologram, with lower center errors.'),
]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v)) + '\n')


def select():
    with (REPO.parent / 'audit/revision-figure9-larger-20261102/pool.pkl').open('rb') as f:
        pool, events, tracks, lookup, arrays, final_keys = pickle.load(f)
    truth = pd.read_parquet(ROOT / 'data/labels/validation/seed260745/particles.parquet',
        columns=['frame', 'particle_id', 'x_px', 'y_px', 'z_um', 'diameter_um'])
    indexed = truth.set_index(['frame', 'particle_id']).sort_index()
    by_event = events.set_index('collision_event_id')
    counts = Counter()
    eligible = []
    candidates = pool.loc[pool.identity_state.eq('identity_consistent') &
        pool.static_score.ge(.75) & pool.base_score.ge(.75)]
    for row in candidates.to_dict('records'):
        v, s, score_t = (int(row[k]) for k in ['vanished_track_id', 'survivor_track_id', 'candidate_frame'])
        if (v, s, score_t) not in final_keys:
            continue
        counts['final_retained_strict_identity_both_scores_ge_0p75'] += 1
        e = by_event.loc[int(row['matched_truth_event_id'])]
        t = int(e.frame)
        if score_t != t:
            continue
        counts['exact_event_frame'] += 1
        vp, sp = lookup.get((t - 1, v), None), lookup.get((t - 1, s), None)
        if {vp, sp} != {e.parent1_id, e.parent2_id}:
            continue
        va, sa = arrays[v], arrays[s]
        if any(t + d not in sa[:, 0] for d in DELTAS):
            continue
        if any(t + d not in va[:, 0] for d in range(-4, 0)) or any(t + d in va[:, 0] for d in [0, 1]):
            continue
        counts['all_ten_actual_observations'] += 1
        crops = []
        for d in DELTAS:
            ss = sa[sa[:, 0] == t + d][0]
            x0, y0 = (int(round(x)) - 64 for x in ss[1:3])
            margin = min(x0, y0, 1024 - (x0 + 128), 1024 - (y0 + 128))
            crops.append(dict(frame=t + d, delta=d, x0=x0, y0=y0, hologram_margin_px=margin))
        if min(p['hologram_margin_px'] for p in crops) < 8:
            continue
        counts['all_six_whole_crops_inside_with_8px_margin'] += 1
        observations = []
        z_gaps = []
        valid = True
        pre_gap = None
        for crop in crops:
            frame, d, x0, y0 = (crop[k] for k in ['frame', 'delta', 'x0', 'y0'])
            ss = sa[sa[:, 0] == frame][0]
            local = [(s, ss, int(sp if d < 0 else e.child_id))]
            if d < 0:
                vv = va[va[:, 0] == frame][0]
                local.append((v, vv, int(vp)))
                z_gaps.append(abs(vv[3] - ss[3]) * 100)
                if d == -4:
                    pre_gap = np.linalg.norm(vv[1:3] - ss[1:3]) - (vv[4] + ss[4]) / 2
            for track_id, p, particle_id in local:
                mapped = lookup.get((frame, track_id), None)
                if mapped is not None and mapped != particle_id:
                    valid = False
                    break
                q = indexed.loc[(frame, particle_id)]
                frame_truth = indexed.xs(frame, level='frame')
                nearest = int(np.hypot(frame_truth.x_px - p[1], frame_truth.y_px - p[2]).idxmin())
                if nearest != particle_id:
                    valid = False
                    break
                radius = q.diameter_um / 20
                clearances = []
                for x, y, r in [(q.x_px, q.y_px, radius), (p[1], p[2], p[4] / 2)]:
                    clearances.append(min(x - x0, y - y0, x0 + 127 - x, y0 + 127 - y) - r)
                if min(clearances) < 3:
                    valid = False
                    break
                xy = float(np.hypot(p[1] - q.x_px, p[2] - q.y_px))
                observations.append(dict(frame=frame, delta=d, track_id=track_id, truth_particle_id=particle_id,
                    x=float(p[1]), y=float(p[2]), z=float(p[3]), diameter_px=float(p[4]),
                    truth_x=float(q.x_px), truth_y=float(q.y_px), truth_diameter_um=float(q.diameter_um),
                    one_to_one_mapping_present=mapped is not None, nearest_truth_particle_is_expected=True,
                    xy_error_px=xy, xy_error_um=xy * 10, relative_radius_error=xy / radius,
                    z_error_um=float((p[3] - 1) * 100 - q.z_um), crop_disk_clearance_px=min(clearances)))
            if not valid:
                break
        if not valid:
            continue
        counts['identity_and_whole_particle_disks'] += 1
        if pre_gap < 1 or max(z_gaps) > 500:
            continue
        counts['separated_pre_disks_and_common_focus'] += 1
        eligible.append({**row, 'truth_event': {'collision_event_id': int(e.name), **e.to_dict()},
            'observations': observations, 'crops': crops,
            'max_xy_error_px': max(p['xy_error_px'] for p in observations),
            'mean_xy_error_px': float(np.mean([p['xy_error_px'] for p in observations])),
            'max_relative_radius_error': max(p['relative_radius_error'] for p in observations),
            'min_parent_diameter_um': float(min(e.parent1_diameter_um, e.parent2_diameter_um)),
            'min_hologram_margin_px': min(p['hologram_margin_px'] for p in crops),
            'pre_edge_gap_px': float(pre_gap), 'max_pre_z_separation_um': float(max(z_gaps)),
            'max_abs_z_error_um': max(abs(p['z_error_um']) for p in observations),
            'final_candidate_retained': True, 'display_deltas': list(DELTAS),
            'display_reference_frame': t, 'scoring_candidate_offset_from_truth_frames': 0})
    for size in [100, 120, 150, 180]:
        for error in [2, 3, 4, 5]:
            counts[f'parents_ge_{size}um_max_xy_le_{error}px'] = sum(
                r['min_parent_diameter_um'] >= size and r['max_xy_error_px'] <= error
                and r['max_relative_radius_error'] <= .4 and r['max_abs_z_error_um'] <= 250
                for r in eligible)
    write('eligible.json', sorted(eligible, key=lambda r: (-r['min_parent_diameter_um'], r['max_xy_error_px'])))
    selected = []
    for case_id, frame, vanished, survivor, reason in PREVIEW_CASES:
        matches = [r for r in eligible if (int(r['candidate_frame']), int(r['vanished_track_id']),
            int(r['survivor_track_id'])) == (frame, vanished, survivor)]
        assert len(matches) == 1
        r = dict(matches[0], review_id=case_id, preview_reason=reason)
        assert r['max_xy_error_px'] <= 5 and r['max_relative_radius_error'] <= .4 and r['max_abs_z_error_um'] <= 250
        selected.append(r)
    write('selected-for-preview.json', selected)
    write('search-protocol.json', dict(counts=counts, crop_size_px=128,
        crop_policy='Actual continuing-track-centered crop; all six full crop squares inside 1024x1024 hologram with >=8px margin. No crop clamping, shifting, synthetic fill, or black padding.',
        identity='Strict identity-consistent and final retained; scoring candidate equals truth event frame; all 10 positions present, nearest expected truth, no conflicting one-to-one mappings; no terminating positions at or after time zero.',
        geometry='Both true and measured disks have >=3px clearance inside crop; initial measured edge gap >=1px; pre-event track Z separation <=500um.',
        selection='Post-hoc figure illustration only. All shown center errors are checked against truth. Performance, models, source measurements and manuscript are unchanged.',
        focus='Continuing-track measured Z; 3 phase-retrieval iterations; original 1st/99th-percentile contrast.'))
    print(json.dumps({k: int(v) for k, v in counts.items()}, indent=2), flush=True)
    for r in sorted(eligible, key=lambda r: (-r['min_parent_diameter_um'], r['max_xy_error_px'])):
        if r['max_xy_error_px'] > 5 or r['max_relative_radius_error'] > .4 or r['max_abs_z_error_um'] > 250:
            continue
        e = r['truth_event']
        print('frame', int(r['candidate_frame']), 'tracks', int(r['vanished_track_id']), int(r['survivor_track_id']),
            'D', round(e['parent1_diameter_um']), round(e['parent2_diameter_um']), round(e['child_diameter_um']),
            'XYmean/max', round(r['mean_xy_error_px'], 2), round(r['max_xy_error_px'], 2),
            'cropmargin', r['min_hologram_margin_px'], 'scores', round(r['static_score'], 3), round(r['base_score'], 3), flush=True)


def render():
    import frame_gallery as gallery
    import frame_gallery_tracks as overlay
    import torch
    from PIL import Image
    from ra3d.acquisition import Acquisition, Reconstructor
    from ra3d._core.track_images import point_at

    selected = json.loads((OUT / 'selected-for-preview.json').read_text())
    tracks = pd.read_parquet(ROOT / 'runs/tracking/seed260745/smoothed.parquet')
    torch.set_num_threads(4)
    acq = Acquisition.load(ROOT / 'config/seed260745-collision.yaml')
    assert acq.phase_iterations == 3
    recon = Reconstructor(acq, 'cpu')
    reviews = []
    for r in selected:
        case_id = r['review_id']
        folder = OUT / case_id
        for sub in ['images', 'tracks', 'cache']:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        gallery.OUT = overlay.OUT = folder
        v = tracks.loc[tracks.track_id.eq(r['vanished_track_id'])].sort_values('frame')
        s = tracks.loc[tracks.track_id.eq(r['survivor_track_id'])].sort_values('frame')
        panels, audits = [], []
        for i, d in enumerate(DELTAS, 1):
            f = r['display_reference_frame'] + d
            point = point_at(s, f, nearest=False)
            assert point is not None
            cache = folder / 'cache' / f'{i}.npz'
            if cache.exists():
                saved = np.load(cache)
                assert np.allclose(saved['center'], [point.x, point.y, point.z])
                raw, valid, x0, y0 = saved['raw'], saved['valid'], int(saved['x0']), int(saved['y0'])
            else:
                with torch.no_grad():
                    plane = recon.intensity(f, point.z)
                raw, x0, y0, valid = original.crop_with_black_padding(plane, point.x, point.y, 128)
                np.savez_compressed(cache, raw=raw, valid=valid, x0=x0, y0=y0, center=[point.x, point.y, point.z])
            assert valid.all() and raw.shape == (128, 128)
            assert x0 == round(point.x) - 64 and y0 == round(point.y) - 64
            margin = min(x0, y0, 1024 - x0 - 128, 1024 - y0 - 128)
            assert margin >= 8
            display = original.normalize_valid_crop(raw, valid)
            name = f'{case_id}-{i}-frame-{f}.png'
            Image.fromarray(display).save(folder / 'images' / name)
            panel = dict(label=f'{case_id}-{i}', filename=name, frame=f, delta=d, time_ms=d / 4,
                x0=x0, y0=y0, center_x=point.x, center_y=point.y, center_z=point.z,
                recorded_fraction=1., hologram_margin_px=margin,
                contrast_low_high=np.percentile(raw, [1, 99]).tolist(), png_sha256=gallery.sha(folder / 'images' / name))
            panels.append(panel)
            audits.append(overlay.render_overlay(panel, v, s))
        e = r['truth_event']
        title = f"{case_id}: {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} -> {e['child_diameter_um']:.0f} um"
        gallery.contact_sheet(title, panels, OUT / f'{case_id}-frames.png', with_tracks=True)
        reviews.append(dict(case_id=case_id, source=r, panels=panels, overlay_audit=audits))
        write('rendered-cases.json', reviews)
        print(title, 'XY max', round(r['max_xy_error_px'], 3), 'crop margin', r['min_hologram_margin_px'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['select', 'render'])
    args = parser.parse_args()
    {'select': select, 'render': render}[args.action]()
