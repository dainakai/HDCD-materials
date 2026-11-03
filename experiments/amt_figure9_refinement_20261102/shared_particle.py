#!/usr/bin/env python3
"""Find the paper's duplicate-track phenomenon using independent physical truth.

This is an illustrative-case audit of frozen test outputs, never a relabeling of
training data or a change in event-evaluation policy. A shared image component
is not evidence that two tracks follow the same physical particle.
"""
import argparse
from collections import Counter
from functools import lru_cache
import json

from search import REPO, ROOT, original
import numpy as np
import pandas as pd

OUT = REPO.parent/'audit/revision-figure9-shared-particle-20261103'
OUT.mkdir(exist_ok=True)
KEY = ['vanished_track_id', 'survivor_track_id', 'candidate_frame']


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v))+'\n')


def select(*, min_drop=0., min_fraction=.8, min_joint=2, prefix=''):
    features = pd.read_parquet(ROOT/'data/features/validation/seed260745/paper92.parquet')
    scores = pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet')
    full = features.merge(scores, on=KEY, validate='one_to_one')
    tracks = pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    arrays = {int(t): g.set_index('frame')[['x','y','z','diameter_px']]
              for t,g in tracks.groupby('track_id', sort=False)}
    truth = pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',
        columns=['frame','particle_id','x_px','y_px','z_um','diameter_um'])
    by_frame = {int(f): g.reset_index(drop=True) for f,g in truth.groupby('frame', sort=False)}
    mapping = pd.read_parquet(ROOT/'runs/tracking/seed260745/mapping.parquet')
    lookup = mapping.set_index(['frame','observed_track_id']).source_particle_id
    events = pd.read_parquet(ROOT/'data/labels/validation/seed260745/collision_events.parquet')
    predictions = pd.read_parquet(ROOT/'results/ours/predictions.parquet')
    final = set(zip(predictions.vanished_track_id, predictions.survivor_track_id, predictions.candidate_frame_raw))

    @lru_cache(maxsize=None)
    def audit_point(tid, frame):
        if frame not in arrays[tid].index:
            return None
        p = arrays[tid].loc[frame]
        gt = by_frame[frame]
        xy = np.hypot(gt.x_px-p.x, gt.y_px-p.y).to_numpy()
        dz = ((p.z-1)*100-gt.z_um).to_numpy()
        # Generous axial tolerance is only an initial screen. Report actual
        # errors and all alternatives so final visual inspection can be stricter.
        plausible = np.flatnonzero((xy <= gt.diameter_um.to_numpy()/20) & (np.abs(dz) <= 250))
        nearest = int(np.argmin(xy))
        mapped = lookup.get((frame,tid),None)
        result = dict(frame=frame, track_id=tid, x=float(p.x), y=float(p.y),
            z_plane=float(p.z), diameter_px=float(p.diameter_px),
            plausible_truth_ids=[int(gt.iloc[i].particle_id) for i in plausible],
            nearest_xy_truth_id=int(gt.iloc[nearest].particle_id),
            one_to_one_source_id=int(mapped) if mapped is not None else None)
        if len(plausible)==1:
            i=int(plausible[0]);q=gt.iloc[i]
            result.update(truth_id=int(q.particle_id),truth_x=float(q.x_px),truth_y=float(q.y_px),
                truth_z_um=float(q.z_um),truth_diameter_um=float(q.diameter_um),
                xy_error_px=float(xy[i]),z_error_um=float(dz[i]),
                mapping_agrees=mapped is None or int(mapped)==int(q.particle_id))
        else:
            result['truth_id']=None
        return result

    counts=Counter(); found=[]
    # Search the full score union, not the old >=7 reconstructed-frame pool.
    # Retain all positive score decreases; size of decrease is reported rather
    # than covertly relaxed. No crop/termination-time prefilter changes identity.
    subset=full.loc[full.static_score.ge(.9)&(full.static_score-full.base_score).gt(min_drop)]
    for r in subset.to_dict('records'):
        v,s,t=(int(r[k]) for k in KEY)
        counts['high_trajectory_score_and_any_decrease']+=1
        obs=[p for f in range(max(0,t-10),min(6000,t+5))
             for tid in [v,s] if (p:=audit_point(tid,f)) is not None]
        pre=[p for p in obs if p['frame']<=t]
        tally={tid:Counter(p['truth_id'] for p in pre if p['track_id']==tid and p['truth_id'] is not None)
               for tid in [v,s]}
        shared=set(tally[v])&set(tally[s])
        if not shared:continue
        counts['some_same_truth_support']+=1
        source=max(shared,key=lambda q:min(tally[v][q],tally[s][q]))
        support=[tally[tid][source] for tid in [v,s]]
        total=[sum(p['track_id']==tid for p in pre) for tid in [v,s]]
        if min(support)<2 or any(n/max(1,nt)<min_fraction for n,nt in zip(support,total)):
            continue
        same=[p for p in obs if p['truth_id']==source]
        joint=[f for f in range(max(0,t-10),t+1)
               if all(any(p['frame']==f and p['track_id']==tid for p in same) for tid in [v,s])]
        if len(joint)<min_joint:continue
        counts['same_physical_particle_repeatedly']+=1
        panels=[]
        for f in range(max(0,t-10),min(6000,t+5)):
            sp=audit_point(s,f)
            if sp is None or sp['truth_id']!=source:continue
            x0,y0=(int(round(sp[k]))-64 for k in ['x','y'])
            margin=min(x0,y0,1024-x0-128,1024-y0-128)
            vp=audit_point(v,f)
            radius=sp['truth_diameter_um']/20
            disk_margin=min(sp['truth_x']-x0,x0+127-sp['truth_x'],sp['truth_y']-y0,y0+127-sp['truth_y'])-radius
            panels.append(dict(frame=f,delta=f-t,x0=x0,y0=y0,hologram_margin_px=margin,
                whole_crop=margin>=0,truth_disk_margin_px=disk_margin,
                red_present=vp is not None,red_same=vp is not None and vp['truth_id']==source,
                blue_observation=sp,red_observation=vp))
        near=events.loc[(events.frame-t).abs().le(5)].copy()
        near['distance_xy_um']=np.hypot(near.x_um-r['x_um'],near.y_um-r['y_um'])
        related=events.loc[events.parent1_id.eq(source)|events.parent2_id.eq(source)|events.child_id.eq(source)]
        r.update(shared_truth_id=source,identity_support=support,pre_observation_count=total,
            simultaneous_shared_frames=joint,observations=obs,panels=panels,
            score_decrease=r['static_score']-r['base_score'],final_retained=(v,s,t) in final,
            no_truth_event_in_common_gate=not near.distance_xy_um.le(350).any(),
            nearest_event_in_time_gate=near.sort_values('distance_xy_um').head(1)[['collision_event_id','frame','distance_xy_um']].to_dict('records'),
            related_truth_events=related[['collision_event_id','frame','parent1_id','parent2_id','child_id']].to_dict('records'))
        found.append(r)
    found.sort(key=lambda r:-r['score_decrease'])
    write(prefix+'all-shared-candidates.json',found)
    summary=[]
    for r in found:
        good=[p for p in r['panels'] if p['whole_crop'] and p['truth_disk_margin_px']>=0
              and (not p['red_present'] or p['red_same'])]
        core=[p['delta'] for p in good if -4<=p['delta']<=1]
        simultaneous=[p['delta'] for p in good if p['red_same']]
        no_red=[p['delta'] for p in good if not p['red_present']]
        q={k:r[k] for k in KEY+['shared_truth_id','static_score','base_score','score_decrease',
            'identity_support','pre_observation_count','final_retained','no_truth_event_in_common_gate']}
        q.update(interior_core_deltas=core,interior_shared_deltas=simultaneous,
            interior_blue_only_deltas=no_red,truth_diameter_um=r['panels'][0]['blue_observation']['truth_diameter_um'])
        summary.append(q)
    write(prefix+'summary.json',summary)
    write(prefix+'search-protocol.json',dict(full_scoring_union=len(full),counts=counts,
        initial_screen=f'trajectory score >=0.9 and decrease >{min_drop} after image features',
        independent_truth_screen=f'A unique physical particle contains the observed center in its projected disk and lies within 250 um axially; >=2 observations per track and >={min_fraction:.0%} of pre/candidate-time observations share one truth ID; >={min_joint} simultaneous shared observations.',
        limitations='This is a documented geometrical screen, not a proof of absence of every duplicate-tracking pattern. Intermittent associations below the stated support threshold remain outside this screen.',
        crop='Fully recorded 128x128 pixels, centered on actual continuing-track positions. Display times considered independently, without changing candidate time or scores.',
        paper_definition='CollisionPaper template.tex lines 164-169 and 914-925: multiple observed IDs assigned to a single physical particle.'))
    print(json.dumps(dict(counts=counts,shared_candidates=len(found)),indent=2),flush=True)
    for q in summary:
        if q['score_decrease']>=.1:print(json.dumps(q),flush=True)


def render():
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import frame_gallery as gallery
    import frame_gallery_tracks as overlay
    from ra3d.acquisition import Acquisition, Reconstructor
    from ra3d._core.track_images import point_at
    rows=json.loads((OUT/'intermittent-all-shared-candidates.json').read_text())
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    acq=Acquisition.load(ROOT/'config/seed260745-collision.yaml')
    assert acq.phase_iterations==3
    torch.set_num_threads(4)
    recon=Reconstructor(acq,'cpu')
    # All presented observations are individually checked against the SAME
    # physical particle. Missing observations are never fabricated.
    choices=[('D1',4838,4819,1908,[-4,-3,-2,-1,0,1]),
             ('D2',18567,18658,5508,[-3,-2,-1,0,1,2]),
             ('D3',10387,10980,3587,[-4,-3,-2,-1,0,1]),
             ('D4',17945,18031,5337,[-6,-5,-4,-2,0,2])]
    rendered=[]
    for cid,vid,sid,t,deltas in choices:
        r=next(r for r in rows if tuple(int(r[k]) for k in KEY)==(vid,sid,t))
        folder=OUT/cid
        for sub in ['images','tracks','proof','cache']:(folder/sub).mkdir(parents=True,exist_ok=True)
        gallery.OUT=overlay.OUT=folder
        v=tracks.loc[tracks.track_id.eq(vid)].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(sid)].sort_values('frame')
        panels=[];audits=[]
        for i,d in enumerate(deltas,1):
            f=t+d
            checked=next(p for p in r['panels'] if p['frame']==f)
            assert checked['whole_crop'] and checked['truth_disk_margin_px']>0
            assert not checked['red_present'] or checked['red_same']
            assert all(p['mapping_agrees'] and p['nearest_xy_truth_id']==r['shared_truth_id']
                for p in [checked['blue_observation'],checked['red_observation']] if p is not None)
            p=point_at(s,f,nearest=False);assert p is not None
            cache=folder/'cache'/f'{i}.npz'
            if cache.exists():
                saved=np.load(cache);assert np.allclose(saved['center'],[p.x,p.y,p.z])
                raw,valid,x0,y0=saved['raw'],saved['valid'],int(saved['x0']),int(saved['y0'])
            else:
                with torch.no_grad():plane=recon.intensity(f,p.z)
                raw,x0,y0,valid=original.crop_with_black_padding(plane,p.x,p.y,128)
            assert valid.all() and (x0,y0)==(checked['x0'],checked['y0'])
            display=original.normalize_valid_crop(raw,valid)
            name=f'{cid}-{i}-frame-{f}.png';Image.fromarray(display).save(folder/'images'/name)
            np.savez_compressed(cache,raw=raw,valid=valid,display=display,center=[p.x,p.y,p.z],x0=x0,y0=y0)
            panel=dict(label=f'{cid}-{i}',filename=name,frame=f,delta=d,x0=x0,y0=y0,
                center_x=p.x,center_y=p.y,center_z=p.z,recorded_fraction=1.,
                hologram_margin_px=checked['hologram_margin_px'],truth_audit=checked,
                png_sha256=gallery.sha(folder/'images'/name))
            panels.append(panel);audits.append(overlay.render_overlay(panel,v,s))
            # Separate audit view only: yellow physical-truth outline/center.
            # The paper-style track images above contain no truth annotations.
            with Image.open(folder/'tracks'/name) as im:
                proof=im.copy();draw=ImageDraw.Draw(proof)
                q=checked['blue_observation'];cx=(q['truth_x']-x0+.5)*2;cy=(q['truth_y']-y0+.5)*2
                rad=q['truth_diameter_um']/10
                draw.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),outline='#edbe38',width=2)
                draw.line((cx-4,cy,cx+4,cy),fill='#edbe38',width=1)
                draw.line((cx,cy-4,cx,cy+4),fill='#edbe38',width=1)
                proof.save(folder/'proof'/name)
            print(f'Rendered {cid} frame {f}, delta {d}, shared truth {r["shared_truth_id"]}',flush=True)
        gallery.contact_sheet(f'{cid}: one true particle, two track IDs',panels,OUT/f'{cid}-frames.png',with_tracks=True)
        rendered.append(dict(case_id=cid,source=r,panels=panels,overlay_audit=audits))
        write('rendered-cases.json',rendered)


def diagnostics():
    """Show excluded/missing frames as diagnostics, never as eligible panels."""
    import torch
    from PIL import Image
    import frame_gallery as gallery
    import frame_gallery_tracks as overlay
    from ra3d.acquisition import Acquisition, Reconstructor
    from ra3d._core.track_images import point_at
    rows=json.loads((OUT/'intermittent-all-shared-candidates.json').read_text())
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    truth=pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet')
    torch.set_num_threads(4)
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cpu')
    result=[]
    for cid,vid,sid,t,deltas in [('D2',18567,18658,5508,[-4]),('D4',17945,18031,5337,[-3,-1,1])]:
        r=next(r for r in rows if tuple(int(r[k]) for k in KEY)==(vid,sid,t))
        folder=OUT/cid;gallery.OUT=overlay.OUT=folder
        v=tracks.loc[tracks.track_id.eq(vid)].sort_values('frame');s=tracks.loc[tracks.track_id.eq(sid)].sort_values('frame')
        for d in deltas:
            f=t+d;actual=point_at(s,f,nearest=False)
            center=[float(np.interp(f,s.frame,s[k])) for k in ['x','y','z']]
            name=f'{cid}-diagnostic-frame-{f}.png'
            cache=folder/'cache'/f'diagnostic-{f}.npz'
            if cache.exists():
                saved=np.load(cache);assert np.allclose(saved['center'],center)
                raw,valid,x0,y0=saved['raw'],saved['valid'],int(saved['x0']),int(saved['y0'])
            else:
                with torch.no_grad():plane=recon.intensity(f,center[2])
                raw,x0,y0,valid=original.crop_with_black_padding(plane,*center[:2],128)
            assert valid.all()
            display=original.normalize_valid_crop(raw,valid)
            Image.fromarray(display).save(folder/'images'/name)
            np.savez_compressed(cache,raw=raw,valid=valid,display=display,center=center,x0=x0,y0=y0)
            panel=dict(case_id=cid,label=f'{cid} diagnostic',filename=name,frame=f,delta=d,x0=x0,y0=y0,
                center_x=center[0],center_y=center[1],center_z=center[2],recorded_fraction=1.,
                center_focus_interpolated=actual is None,eligible_for_paper=False,
                png_sha256=gallery.sha(folder/'images'/name))
            q=truth.loc[truth.frame.eq(f)&truth.particle_id.eq(r['shared_truth_id'])].iloc[0]
            evidence=[]
            for tid,tr in [(vid,v),(sid,s)]:
                p=point_at(tr,f,nearest=False)
                if p is None:
                    evidence.append(dict(track_id=tid,present=False));continue
                xy=float(np.hypot(p.x-q.x_px,p.y-q.y_px))
                evidence.append(dict(track_id=tid,present=True,shared_truth_id=int(q.particle_id),
                    xy_error_to_shared_truth_px=xy,truth_radius_px=float(q.diameter_um/20),
                    z_error_to_shared_truth_um=float((p.z-1)*100-q.z_um),center_inside_truth_disk=xy<=q.diameter_um/20))
            panel['observations']=evidence;panel['overlay_audit']=overlay.render_overlay(panel,v,s)
            result.append(panel)
    write('diagnostic-frames.json',result)
    print(json.dumps({'diagnostic_panels':len(result),'interpolated_center_only':sum(p['center_focus_interpolated'] for p in result)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select','intermittent','render','diagnostics'])
    args=parser.parse_args()
    if args.action=='intermittent':
        select(min_drop=.2,min_fraction=.3,min_joint=1,prefix='intermittent-')
    elif args.action=='render':render()
    elif args.action=='diagnostics':diagnostics()
    else:select()
