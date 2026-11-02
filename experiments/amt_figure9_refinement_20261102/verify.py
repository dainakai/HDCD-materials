#!/usr/bin/env python3
"""Check final display provenance, saved candidate scores and original geometry."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from search import REPO, ROOT, OUT, HERE, original
import numpy as np
import pandas as pd
from ra3d.config import KEY_COLUMNS
from ra3d._core.track_images import point_at
sys.path.append('/home/dai-server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
import pdfplumber


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    selection = load(OUT/'selection.json')
    assert selection['status'] == 'PASS'
    assert sha(OUT/'fig09.pdf') == selection['final_pdf_sha256']
    assert sha(OUT/'fig09.png') == selection['final_png_sha256']
    paper = REPO.parent/'CollisionPaper'
    assert sha(paper/'本番フォーマット/template.tex') == '1fbbcfcf35f5011067d92429359fd7785f98653ace260b49622b32f60e1e83b5'
    assert sha(HERE/'source/original_plot.py') == selection['original_style_source_sha256'] == sha(selection['original_style_source'])
    assert sha(HERE/'source/prepare_figures.py') == selection['submission_preparer_sha256']
    frozen = REPO/'experiments/amt_full_20261101/results'
    checks = load(frozen/'manifest.json')['numerical_outputs_manifest']
    assert len(checks) == 34
    for name, digest in checks.items():
        assert sha(frozen/name) == digest, name
    for field, path in [
        ('reconstruction_config_sha256', 'config/seed260745-collision.yaml'),
        ('candidate_features_sha256', 'data/features/validation/seed260745/paper92.parquet'),
        ('saved_scores_sha256', 'results/ours/candidate_scores.parquet'),
        ('saved_predictions_sha256', 'results/ours/predictions.parquet'),
    ]:
        assert sha(ROOT/path) == selection[field], field

    features = pd.read_parquet(ROOT/'data/features/validation/seed260745/paper92.parquet',
                               columns=KEY_COLUMNS+['static_score','pre_same_interp_fraction','x_um','y_um'])
    scores = pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet')
    predictions = pd.read_parquet(ROOT/'results/ours/predictions.parquet')
    final_keys = set(zip(predictions.vanished_track_id, predictions.survivor_track_id, predictions.candidate_frame_raw))
    tracks = pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    mapping = pd.read_parquet(ROOT/'runs/tracking/seed260745/mapping.parquet')
    truth = pd.read_parquet(ROOT/'data/labels/validation/seed260745/collision_events.parquet')
    identity = load(OUT/'identity-audit.json')
    comparison = load(OUT/'original-comparison.json')
    assert identity['status'] == comparison['status'] == 'PASS'
    assert sha(ROOT/'runs/tracking/seed260745/smoothed.parquet') == identity['track_sha256']
    assert sha(ROOT/'runs/tracking/seed260745/mapping.parquet') == identity['mapping_sha256']
    assert sha(ROOT/'data/labels/validation/seed260745/collision_events.parquet') == identity['truth_events_sha256']

    def candidate(df, case):
        selected = df
        for key in KEY_COLUMNS:
            selected = selected.loc[selected[key].eq(case[key])]
        assert len(selected) == 1
        return selected.iloc[0]

    for case in selection['cases']:
        row = candidate(features, case)
        assert float(row.static_score) == case['trajectory_score']
        assert float(candidate(scores, case).base_score) == case['image_feature_score']
        assert float(row.pre_same_interp_fraction) == case['image_pre_same_fraction']
        key = (case['vanished_track_id'], case['survivor_track_id'], case['candidate_frame'])
        assert (key in final_keys) == case['final_candidate_retained']
        assert case['display_frames'] == [case['candidate_frame']+d for d in [-4,0,1]]
        assert case['display_time_ms'] == [-1.,0.,.25]
        v = tracks.loc[tracks.track_id.eq(case['vanished_track_id'])].sort_values('frame')
        s = tracks.loc[tracks.track_id.eq(case['survivor_track_id'])].sort_values('frame')
        cache = np.load(Path(selection.get('candidate_search_directory',OUT))/f"cache/{case['review_id']}.npz")
        panels = [p for p in selection['panels'] if p['frame'] in case['display_frames']]
        assert len(panels) == 3
        for panel in panels:
            d = panel['delta']; frame = panel['frame']
            raw, display, valid = cache[f'raw_{d}'], cache[f'display_{d}'], cache[f'valid_{d}']
            assert raw.shape == display.shape == valid.shape == (128,128)
            assert panel['recorded_fraction'] == valid.mean()
            yy, xx = np.indices((128,128))
            xx += int(panel['x0']); yy += int(panel['y0'])
            assert np.array_equal(valid,(xx>=0)&(xx<1024)&(yy>=0)&(yy<1024))
            assert not display[~valid].any()
            assert hashlib.sha256(raw.tobytes()).hexdigest() == panel['raw_array_sha256']
            assert hashlib.sha256(display.tobytes()).hexdigest() == panel['normalized_array_sha256']
            assert hashlib.sha256(valid.tobytes()).hexdigest() == panel['valid_mask_sha256']
            assert np.array_equal(original.normalize_valid_crop(raw, valid), display)
            assert np.allclose(np.percentile(raw[valid],[1,99]), panel['contrast_low_high'])
            sp = point_at(s,frame,nearest=False); vp = point_at(v,frame,nearest=False)
            assert sp is not None
            for axis in 'xyz':
                assert np.isclose(float(s.loc[s.frame.eq(frame),axis].iloc[0]), panel['center_'+axis])
            for axis in 'xy':
                assert panel[axis+'0'] == round(panel['center_'+axis])-64
                assert abs(panel['center_'+axis]-panel[axis+'0']-64) <= .5
            for track in [v,s]:
                actual=track.loc[track.frame.eq(frame)]
                if actual.empty:continue
                p=actual.iloc[0]
                clearance=min(p.x-panel['x0'],panel['x0']+127-p.x,p.y-panel['y0'],panel['y0']+127-p.y)-p.diameter_px/2
                assert clearance>=1, (case['review_id'],frame,clearance)
            assert panel['actual_marker_observations'] == {'terminating':vp is not None,'continuing':True}
            if d < 0 or (case['true_event_id'] < 0 and d == 0): assert vp is not None
            if d > 0: assert vp is None
        if case['true_event_id'] >= 0:
            e = truth.loc[truth.planned_event_id.eq(case['true_event_id'])]
            assert len(e) == 1 and int(e.iloc[0].frame) == case['candidate_frame']
            for name in ['parent1_diameter_um','parent2_diameter_um','child_diameter_um']:
                assert float(e.iloc[0][name]) == case['truth_event'][name]
        else:
            assert case['trajectory_score'] >= .9
            assert case['trajectory_score']-case['image_feature_score'] >= .3
            assert not case['final_candidate_retained']
            t = case['candidate_frame']
            nearby = truth.loc[(truth.frame-t).abs().le(5)]
            assert not np.hypot(nearby.x_um-row.x_um, nearby.y_um-row.y_um).le(350).any()
            support = mapping.loc[mapping.observed_track_id.isin(key[:2]) & mapping.frame.between(t-10,t+1)]
            assert set(support.source_particle_id) == {case['shared_truth_particle_id']}
            assert set(support.observed_track_id) == {case['survivor_track_id']}
        audit = next(c for c in identity['cases'] if c['review_id']==case['review_id'])
        assert audit['strict_training_identity_state'] == case['strict_training_identity_state']
        for observation in audit['observations']:
            assert observation['projected_center_error_px'] <= observation['truth_radius_px']
            assert abs(observation['depth_error_um']) < 100
            assert observation['nearest_truth_center_is_expected']

    geometry = []
    used_fonts = set()
    for path in [paper/'本番フォーマット/figures/fig09.pdf', OUT/'fig09.pdf']:
        with pdfplumber.open(path) as doc:
            page = doc.pages[0]
            boxes = sorted(page.images, key=lambda box:(round(box['top']),box['x0']))
            assert len(boxes) == 6
            geometry.append([[box[k] for k in ['x0','x1','top','bottom']] for box in boxes])
            assert all(-.1 <= c['x0'] <= c['x1'] <= page.width+.1 and -.1 <= c['top'] <= c['bottom'] <= page.height+.1 for c in page.chars)
            if path == OUT/'fig09.pdf':
                used_fonts = {c['fontname'] for c in page.chars}
    assert np.allclose(*geometry,atol=.01), geometry
    fonts = subprocess.check_output(['/usr/bin/pdffonts', str(OUT/'fig09.pdf')], text=True)
    # The unmodified submission relabeler registers an unused default Helvetica.
    # Verify that every font actually used by rendered text is embedded.
    for name in used_fonts:
        rows = [row for row in fonts.splitlines()[2:] if row.split()[0] == name]
        assert len(rows) == 1 and ' yes ' in rows[0], (name, rows)
    assert 'LiberationSans' in fonts
    result = {'status':'PASS', 'final_pdf_sha256':sha(OUT/'fig09.pdf'),
              'selection_sha256':sha(OUT/'selection.json'), 'frozen_evaluation_files_unchanged':len(checks),
              'original_script_identical':True,'original_image_panel_geometry_identical':True,
              'used_fonts_embedded_and_no_clipped_text':True,'used_fonts':sorted(used_fonts),
              'unused_default_helvetica_resource':True,'display_crops_checked':6,'crop_size_px':128,
              'original_percentile_normalization_exact':True,'actual_track_observations_checked':True,
              'target_centered_original_crop_policy':True,'black_padding_excluded_from_percentiles':True,
              'fully_recorded_panels':sum(p['recorded_fraction']==1 for p in selection['panels']),
              'measured_particle_disks_clear_crop_boundaries':True,
              'candidate_scores_and_final_policy_checked':True,'artifact_trajectory_score_above_0p9':True,
              'truth_geometry_observations_checked':sum(len(c['observations']) for c in identity['cases']),
              'identity_audit_sha256':sha(OUT/'identity-audit.json'),
              'original_comparison_sha256':sha(OUT/'original-comparison.json'),
              'artifact_has_no_truth_in_common_evaluation_gate':True,'manuscript_unchanged':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'results').mkdir(exist_ok=True)
    shutil.copyfile(OUT/'verification.json',HERE/'results/verification.json')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
