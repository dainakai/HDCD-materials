#!/usr/bin/env python3
"""Audit displayed observations independently of the one-to-one training labels."""
import hashlib
import json
from pathlib import Path
from search import REPO, ROOT
import numpy as np
import pandas as pd

OUT = REPO.parent/'audit/revision-figure9-centered-20261102'
OLD = Path('/media/dai-server/DATAM2/repos/260727/synth/collision_benchmark_260715_200p_6000f_clean_2000coll_halfmix25_v1_outside256_yjitter10_seed260745')
ORIGINAL = Path('/media/dai-server/DATAM2/siderepos/260804/CollisionPaper/figures/image_feature_case_study/figure_provenance.json')
CHOSEN = ('C04', 'H01')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x))+'\n')


def main():
    rows = json.loads((OUT/'shortlist.json').read_text())
    tracks = pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    mapping = pd.read_parquet(ROOT/'runs/tracking/seed260745/mapping.parquet')
    events = pd.read_parquet(ROOT/'data/labels/validation/seed260745/collision_events.parquet')
    cases = []
    for name in CHOSEN:
        r = next(r for r in rows if r['review_id'] == name)
        t = int(r['candidate_frame'])
        ids = [int(r['vanished_track_id']), int(r['survivor_track_id'])]
        e = r.get('truth_event')
        observations = []
        for d in r['display_deltas']:
            frame = t+d
            gt = pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',
                                 filters=[('frame', '=', frame)])
            actual = tracks.loc[tracks.track_id.isin(ids) & tracks.frame.eq(frame)]
            for _, p in actual.iterrows():
                expected = (e['parent1_id'] if p.track_id == ids[0] else e['parent2_id']) if e and d < 0 else (e['child_id'] if e else 3605)
                truth = gt.loc[gt.particle_id.eq(expected)]
                assert len(truth) == 1
                q = truth.iloc[0]
                distances = np.hypot(gt.x_px-p.x, gt.y_px-p.y)
                assert int(gt.loc[distances.idxmin()].particle_id) == expected
                xy_px = float(np.hypot(q.x_px-p.x, q.y_px-p.y))
                radius_px = float(q.diameter_um/20)
                # Track z is a one-based reconstruction plane, unlike truth z_slice.
                dz_um = float((p.z-1)*100-q.z_um)
                assert xy_px <= radius_px and abs(dz_um) < 100
                # Every target disk is inside the recorded image, including post-frame.
                assert min(q.x_px, q.y_px, 1023-q.x_px, 1023-q.y_px) > radius_px
                observations.append(dict(frame=frame, track_id=int(p.track_id), truth_particle_id=int(expected),
                    track_xy_px=[float(p.x),float(p.y)], truth_xy_px=[float(q.x_px),float(q.y_px)],
                    projected_center_error_px=xy_px, truth_radius_px=radius_px,
                    depth_error_um=dz_um, truth_diameter_um=float(q.diameter_um),
                    nearest_truth_center_is_expected=True, projected_center_inside_truth_disk=True))
        support = mapping.loc[mapping.observed_track_id.isin(ids) & mapping.frame.between(t-10,t+1)]
        case = dict(review_id=name, observations=observations, strict_training_identity_state=r['identity_state'],
            one_to_one_mapping=support.to_dict('records'),
            note='Display-only geometric audit, not a change to training labels or event evaluation. One-to-one mapping cannot assign both duplicated IDs to the same truth particle in a frame.')
        if not e:
            near = events.loc[(events.frame-t).abs().le(5)].copy()
            near['distance_xy_um'] = np.hypot(near.x_um-r['x_um'], near.y_um-r['y_um'])
            nearest = near.sort_values('distance_xy_um').iloc[0]
            assert nearest.distance_xy_um > 350
            assert r['static_score'] >= .9 and r['static_score']-r['base_score'] >= .3
            assert not r['final_candidate_retained']
            assert set(support.source_particle_id) == {3605}
            case.update(shared_truth_particle_id=3605, truth_diameter_um=observations[0]['truth_diameter_um'],
                no_truth_in_common_evaluation_gate=True,
                nearest_event_in_time_gate=nearest[['planned_event_id','frame','distance_xy_um']].to_dict(),
                nearby_event_note='The displayed particle is a parent of event 757 two frames later; the candidate lies 481 um from that event, outside the common 350 um gate. The artifact is the duplicate pair of track IDs on particle 3605, not a claim that no collision occurs elsewhere in the view.')
        cases.append(case)
    write('identity-audit.json', dict(status='PASS', cases=cases,
        track_sha256=sha(ROOT/'runs/tracking/seed260745/smoothed.parquet'),
        mapping_sha256=sha(ROOT/'runs/tracking/seed260745/mapping.parquet'),
        truth_events_sha256=sha(ROOT/'data/labels/validation/seed260745/collision_events.parquet')))

    provenance = json.loads(ORIGINAL.read_text())
    old_parameters = json.loads((OLD/'metadata/parameters.json').read_text())
    generation = REPO.parent/'audit/revision-preflight-20261030/generation'
    new_parameters = json.loads((generation/'seed260745/metadata/parameters.json').read_text())
    distribution_hashes = {}
    for name in ['diameter_density_csv','velocity_quantiles_csv']:
        before, after = sha(old_parameters[name]), sha(new_parameters[name])
        assert before == after
        distribution_hashes[name] = before
    regenerated_sequences = []
    for seed in range(260740,260746):
        old_dir = OLD.with_name(OLD.name.replace('260745',str(seed)))
        new_dir = generation/f'seed{seed}'
        before, after = sha(old_dir/'labels/collision_events.csv'), sha(new_dir/'labels/collision_events.csv')
        assert before != after
        regenerated_sequences.append(dict(seed=seed, role='test' if seed==260745 else 'training',
            old_events_sha256=before, regenerated_events_sha256=after))
    old_events = pd.read_csv(OLD/'labels/collision_events.csv')
    fields = ['planned_event_id','frame','parent1_id','parent2_id','child_id','x_um','y_um','z_um',
              'parent1_diameter_um','parent2_diameter_um','child_diameter_um']
    old = old_events.loc[old_events.planned_event_id.eq(1165)].iloc[0]
    new = events.loc[events.planned_event_id.eq(1165)].iloc[0]
    assert old.frame == new.frame == 3859 and old.parent1_id != new.parent1_id
    assert not np.allclose(old[['x_um','y_um','z_um']].astype(float), new[['x_um','y_um','z_um']].astype(float))
    write('original-comparison.json', dict(status='PASS', original_provenance_sha256=sha(ORIGINAL),
        generation_depth_um=dict(original=old_parameters['depth_um'],current=new_parameters['depth_um']),
        identical_empirical_input_csv_sha256=distribution_hashes, regenerated_sequences=regenerated_sequences,
        original_cases=provenance['cases'], original_crop_policy=provenance['display_crop_policy'],
        original_normalization=provenance['normalization'],
        original_event_csv_sha256=sha(OLD/'labels/collision_events.csv'),
        current_event_parquet_sha256=sha(ROOT/'data/labels/validation/seed260745/collision_events.parquet'),
        same_planned_slot_1165=dict(original=old[fields].to_dict(),current=new[fields].to_dict(),
            inspected_current_case='C01 / 12486_12575_3859',
            decision='Inspected the regenerated realization of the original planned slot. Selected C04 instead: larger parents and daughter, a visible two-to-one transition, and no continuing duplicate red observation in the post-frame.'),
        dataset_relation='Same training/test seed allocation, byte-identical empirical diameter and velocity distributions, and common HoloD3 measurement models; all five training sequences and the test sequence were regenerated. Generation depth changed from 102.4 to 102.3 mm and the particle realizations differ. Same seed or shared HoloD3 does not imply identical input samples or track IDs.',
        correction='The previous larger-particle version added an incorrect crop-center clamp to avoid padding. This shifted targets to crop edges. Restore target-centered 128-pixel crops and black unrecorded padding; normalize only recorded pixels as in the original.',
        selected_cases=list(CHOSEN), display_frames_relative=[-4,0,1],
        display_time_note='The +0.25 ms frame retains actual observations and whole target disks before they leave the recording area. No post-exit positions are imputed. The original figure used +1.00 ms.',
        scope='Only illustrative selection and display change; current model fits, scores and evaluation metrics remain frozen.'))

    write('search-protocol.json', dict(status='PASS', selected=list(CHOSEN),
        selection='Post-hoc illustrative choice at the author request, not an estimate of an average image-feature effect.',
        inspected_centered_candidates=[r['review_id'] for r in rows],
        artifact_rule='Trajectory score >=0.9, saved image-ensemble score decreases by >=0.3, outside the common truth gate, absent from final policy; shared-particle interpretation audited from actual observations and independent truth geometry.',
        true_rule='Final-policy retained, both scores >=0.9, exact event time, two actual pre-event track observations on the two truth parents, and actual daughter observations at/after the event.',
        crop='128 pixels centered on continuing-track observations; original black padding and recorded-pixel 1st–99th percentile normalization.',
        time_frames=[-4,0,1], focus='Actual continuing-track smoothed depth; no truth-based refocusing.',
        historical_searches=['search.py: smaller centered examples','larger.py: superseded off-center examples'],
        strict_training_label_note='C04 has identity_unknown in the strict training-label audit because the terminating ID lacks a one-to-one truth match. Its displayed observations are checked independently against the correct parents and daughter. Do not claim strict identity_consistent.'))
    print(json.dumps({'status':'PASS','selected':list(CHOSEN),'observations_checked':sum(len(c['observations']) for c in cases)}))


if __name__ == '__main__':
    main()
