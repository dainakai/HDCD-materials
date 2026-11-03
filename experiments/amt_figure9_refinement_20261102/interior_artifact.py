#!/usr/bin/env python3
"""Audit and show interior false candidates without relabeling early real collisions."""
import argparse
import json
import pickle
from collections import Counter

from search import REPO, ROOT, original
import numpy as np
import pandas as pd

OUT = REPO.parent/'audit/revision-figure9-interior-artifact-20261103'
OUT.mkdir(exist_ok=True)


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v))+'\n')


def select():
    with (REPO.parent/'audit/revision-figure9-larger-20261102/pool.pkl').open('rb') as f:
        pool, events, tracks, lookup, arrays, final_keys = pickle.load(f)
    truth = pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',
        columns=['frame','particle_id','x_px','y_px','z_um','diameter_um'])
    by_frame = {int(f): g for f, g in truth.groupby('frame')}
    counts = Counter()
    eligible, rejected = [], []
    subset = pool.loc[pool.static_score.ge(.9) & (pool.static_score-pool.base_score).ge(.2) & pool.spatial_positive.eq(0)]
    for r in subset.to_dict('records'):
        v,s,t = (int(r[k]) for k in ['vanished_track_id','survivor_track_id','candidate_frame'])
        counts['trajectory_ge_0p9_drop_ge_0p2_spatial_negative'] += 1
        if (v,s,t) in final_keys:
            continue
        counts['absent_from_final_policy'] += 1
        va,sa = arrays[v],arrays[s]
        if not all(t+d in sa[:,0] for d in [-1,0,1]) or not all(t+d in va[:,0] for d in [-1,0]):
            continue
        counts['actual_core_observations'] += 1
        deltas = next((list(ds) for ds in [range(-4,2),range(-3,3)] if all(t+d in sa[:,0] for d in ds)), None)
        if deltas is None:
            continue
        observations,panels = [],[]
        valid = True
        for d in deltas:
            frame=t+d
            sp=sa[sa[:,0]==frame][0]
            x0,y0=(int(round(p))-64 for p in sp[1:3])
            margin=min(x0,y0,1024-x0-128,1024-y0-128)
            if margin<8:
                valid=False
                break
            panels.append(dict(frame=frame,delta=d,x0=x0,y0=y0,hologram_margin_px=margin))
            gt=by_frame[frame]
            for tid,a in [(v,va),(s,sa)]:
                ps=a[a[:,0]==frame]
                if not len(ps):
                    continue
                p=ps[0]
                distance=np.hypot(gt.x_px-p[1],gt.y_px-p[2])
                q=gt.loc[distance.idxmin()]
                xy=float(distance.min());dz=float((p[3]-1)*100-q.z_um)
                radius=float(q.diameter_um/20)
                clearance=min(p[1]-x0,x0+127-p[1],p[2]-y0,y0+127-p[2])-p[4]/2
                truth_clearance=min(q.x_px-x0,x0+127-q.x_px,q.y_px-y0,y0+127-q.y_px)-radius
                if clearance<1 or truth_clearance<1 or xy>radius or abs(dz)>250:
                    valid=False
                    break
                mapped=lookup.get((frame,tid),None)
                if mapped is not None and int(mapped)!=int(q.particle_id):
                    valid=False
                    break
                observations.append(dict(frame=frame,delta=d,track_id=tid,
                    x=float(p[1]),y=float(p[2]),z=float(p[3]),diameter_px=float(p[4]),
                    truth_particle_id=int(q.particle_id),truth_x=float(q.x_px),truth_y=float(q.y_px),
                    truth_z_um=float(q.z_um),truth_diameter_um=float(q.diameter_um),
                    xy_error_px=xy,depth_error_um=dz,truth_radius_px=radius,
                    nearest_truth_center_is_expected=True,one_to_one_mapping_present=mapped is not None))
            if not valid:
                break
        if not valid:
            continue
        counts['whole_six_crops_and_geometric_identity'] += 1
        ids_by_track={tid:sorted({p['truth_particle_id'] for p in observations if p['track_id']==tid}) for tid in [v,s]}
        if any(len(ids)!=1 for ids in ids_by_track.values()):
            continue
        vsource,ssource=ids_by_track[v][0],ids_by_track[s][0]
        related=events.loc[events.parent1_id.isin([vsource,ssource]) & events.parent2_id.isin([vsource,ssource])]
        if len(related):
            rejected.append(dict(candidate_frame=t,vanished_track_id=v,survivor_track_id=s,
                trajectory_score=r['static_score'],image_feature_score=r['base_score'],
                reason='Same two true parents collide at a different scheduled time; not a pure shared-particle or crossing artifact.',
                related_events=related[['collision_event_id','frame','parent1_id','parent2_id']].to_dict('records')))
            counts['excluded_different_time_of_real_collision'] += 1
            continue
        nearby=events.loc[(events.frame-t).abs().le(5)].copy()
        nearby['xy_distance_um']=np.hypot(nearby.x_um-r['x_um'],nearby.y_um-r['y_um'])
        assert not nearby.xy_distance_um.le(350).any()
        r.update(kind='shared_particle_artifact' if vsource==ssource else 'projected_crossing_artifact',
            truth_ids_by_track=ids_by_track,observations=observations,panels=panels,
            display_deltas=deltas,missing_minus1ms_observation=t-4 not in sa[:,0],
            min_hologram_margin_px=min(p['hologram_margin_px'] for p in panels),
            score_decrease=float(r['static_score']-r['base_score']),final_candidate_retained=False,
            nearest_event_in_time_gate=nearby.sort_values('xy_distance_um').head(1)[['collision_event_id','frame','xy_distance_um']].to_dict('records'))
        paired=[];projected_gaps=[]
        for d in deltas:
            ps=[p for p in observations if p['delta']==d]
            if len(ps)==2:
                paired.append(abs(ps[0]['truth_z_um']-ps[1]['truth_z_um']))
                projected_gaps.append(float(np.hypot(ps[0]['truth_x']-ps[1]['truth_x'],ps[0]['truth_y']-ps[1]['truth_y'])-ps[0]['truth_radius_px']-ps[1]['truth_radius_px']))
        r['truth_z_separation_um_range']=[min(paired),max(paired)]
        r['truth_projected_edge_gap_px_range']=[min(projected_gaps),max(projected_gaps)]
        r['kind']=('shared_particle_artifact' if vsource==ssource else
            'projected_overlap_artifact' if min(projected_gaps)<0 else 'unrelated_particle_pair_artifact')
        eligible.append(r)
    # One visual candidate per actual pair; keep the largest score drop.
    selected=[];seen=set()
    for r in sorted(eligible,key=lambda r:-r['score_decrease']):
        pair=tuple(sorted(int(p) for ids in r['truth_ids_by_track'].values() for p in ids))
        if pair in seen:
            continue
        seen.add(pair)
        r['review_id']=f'B{len(selected)+1:02d}'
        selected.append(r)
    write('eligible.json',eligible)
    write('selected.json',selected)
    write('excluded-real-collision-timing.json',rejected)
    write('search-protocol.json',dict(counts=counts,distinct_examples=len(selected),
        required_trajectory_score=.9,screening_min_score_drop=.2,
        score_drop_note='Prefer >=0.3; retain additional 0.2–0.3 alternatives with exact score changes disclosed. This does not change inference or evaluation.',
        display='Six consecutive actual continuing-track observations, preferably −1 to +0.25ms; if −1ms is missing use −0.75 to +0.5ms. Core −0.25/0/+0.25ms retained.',
        crop='Actual continuing-track-centered 128px crops, all fully recorded with >=8px hologram margin; all measured and truth target disks within the crop.',
        evidence='Stable nearest true particle per track, projected center inside its disk, axial error <=250um, no conflicting one-to-one mappings; classify same-particle vs distinct-particle projection overlap explicitly. Exclude pairs with an actual scheduled collision at another time.',
        evaluation='Absent final policy and no truth event in common ±5frame/350um XY gate. Scores and models are frozen.'))
    print(json.dumps({'counts':counts,'distinct_examples':len(selected)},indent=2),flush=True)
    for r in selected:
        print(r['review_id'],r['kind'],int(r['candidate_frame']),int(r['vanished_track_id']),int(r['survivor_track_id']),
            'scores',r['static_score'],r['base_score'],'margin',r['min_hologram_margin_px'],
            'truthZgap',r['truth_z_separation_um_range'],'deltas',r['display_deltas'],flush=True)


def render():
    import frame_gallery as gallery
    import frame_gallery_tracks as overlay
    import torch
    from PIL import Image
    from ra3d.acquisition import Acquisition, Reconstructor
    from ra3d._core.track_images import point_at

    rows=json.loads((OUT/'selected.json').read_text())
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    acq=Acquisition.load(ROOT/'config/seed260745-collision.yaml')
    assert acq.phase_iterations==3
    torch.set_num_threads(4)
    recon=Reconstructor(acq,'cpu')
    rendered=[]
    for r in rows:
        cid=r['review_id'];folder=OUT/cid
        for sub in ['images','tracks','cache']:(folder/sub).mkdir(parents=True,exist_ok=True)
        gallery.OUT=overlay.OUT=folder
        v=tracks.loc[tracks.track_id.eq(r['vanished_track_id'])].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(r['survivor_track_id'])].sort_values('frame')
        panels=[];audits=[]
        for i,d in enumerate(r['display_deltas'],1):
            f=int(r['candidate_frame'])+d;point=point_at(s,f,nearest=False)
            assert point is not None
            cache=folder/'cache'/f'{i}.npz'
            if cache.exists():
                saved=np.load(cache)
                assert np.allclose(saved['center'],[point.x,point.y,point.z])
                raw,valid,x0,y0=saved['raw'],saved['valid'],float(saved['x0']),float(saved['y0'])
            else:
                with torch.no_grad():plane=recon.intensity(f,point.z)
                raw,x0,y0,valid=original.crop_with_black_padding(plane,point.x,point.y,128)
            assert valid.all()
            display=original.normalize_valid_crop(raw,valid)
            name=f'{cid}-{i}-frame-{f}.png'
            Image.fromarray(display).save(folder/'images'/name)
            np.savez_compressed(folder/'cache'/f'{i}.npz',raw=raw,valid=valid,display=display,
                center=[point.x,point.y,point.z],x0=x0,y0=y0)
            panel=dict(label=f'{cid}-{i}',filename=name,frame=f,delta=d,x0=x0,y0=y0,
                center_x=point.x,center_y=point.y,center_z=point.z,recorded_fraction=1.,
                png_sha256=gallery.sha(folder/'images'/name))
            panels.append(panel);audits.append(overlay.render_overlay(panel,v,s))
        title=f"{cid}: {r['static_score']:.3f} -> {r['base_score']:.3f}"
        gallery.contact_sheet(title,panels,OUT/f'{cid}-frames.png',with_tracks=True)
        rendered.append(dict(case_id=cid,source=r,panels=panels,overlay_audit=audits))
        write('rendered-cases.json',rendered)
        print('Rendered',title,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select','render'])
    args=parser.parse_args();{'select':select,'render':render}[args.action]()
