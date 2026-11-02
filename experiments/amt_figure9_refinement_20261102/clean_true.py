#!/usr/bin/env python3
"""Select illustrative true collisions with accurate measured positions in all frames."""
import json
import pickle
from collections import Counter

from search import REPO, ROOT
import numpy as np
import pandas as pd

OUT=REPO.parent/'audit/revision-figure9-clean-true-20261102'
OUT.mkdir(exist_ok=True)
DELTAS=range(-4,2)


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,
        default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')


def main():
    with (REPO.parent/'audit/revision-figure9-larger-20261102/pool.pkl').open('rb') as f:
        pool,events,tracks,lookup,arrays,final_keys=pickle.load(f)
    truth=pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',
        columns=['frame','particle_id','x_px','y_px','z_um','diameter_um']).set_index(['frame','particle_id']).sort_index()
    by_event=events.set_index('collision_event_id')
    candidates=pool.loc[pool.identity_state.eq('identity_consistent')&pool.static_score.ge(.75)&pool.base_score.ge(.75)]
    counts=Counter();eligible=[]
    for row in candidates.to_dict('records'):
        v,s,score_t=(int(row[k]) for k in ['vanished_track_id','survivor_track_id','candidate_frame'])
        if (v,s,score_t) not in final_keys:continue
        counts['retained_strict_identity_both_scores_ge_0p75']+=1
        e=by_event.loc[int(row['matched_truth_event_id'])]
        t=int(e.frame)
        if score_t==t:counts['score_candidate_at_exact_event_frame']+=1
        # Display time zero is the physical collision, regardless of a +/-1-frame
        # scoring-candidate offset. Preserve and disclose that offset explicitly.
        if min(e.parent1_diameter_um,e.parent2_diameter_um)<150:continue
        counts['both_true_parents_ge_150um']+=1
        vp=lookup.get((t-1,v),np.nan);sp=lookup.get((t-1,s),np.nan)
        if {vp,sp}!={e.parent1_id,e.parent2_id}:continue
        va,sa=arrays[v],arrays[s]
        if any(t+d not in sa[:,0] for d in DELTAS):continue
        if any(t+d not in va[:,0] for d in range(-4,0)):continue
        if any(t+d in va[:,0] for d in [0,1]):continue
        counts['all_observations_no_post_event_terminating_id']+=1
        observations=[];valid=True;z_gaps=[];pre_gap=None
        for d in DELTAS:
            frame=t+d;ss=sa[sa[:,0]==frame][0]
            x0,y0=np.rint(ss[1:3])-64
            local=[(s,ss,int(sp if d<0 else e.child_id))]
            if d<0:
                vv=va[va[:,0]==frame][0];local.append((v,vv,int(vp)))
                z_gaps.append(abs(vv[3]-ss[3])*100)
                if d==-4:
                    pre_gap=np.linalg.norm(vv[1:3]-ss[1:3])-(vv[4]+ss[4])/2
            for track_id,p,particle_id in local:
                mapped=lookup.get((frame,track_id),None)
                if mapped is not None and mapped!=particle_id:
                    valid=False;break
                q=truth.loc[(frame,particle_id)]
                frame_truth=truth.xs(frame,level='frame')
                nearest=int(np.hypot(frame_truth.x_px-p[1],frame_truth.y_px-p[2]).idxmin())
                if nearest!=particle_id:
                    valid=False;break
                xy=float(np.hypot(p[1]-q.x_px,p[2]-q.y_px));radius=q.diameter_um/20
                # Both the true disk and measured disk must be visible.
                for x,y,r in [(q.x_px,q.y_px,radius),(p[1],p[2],p[4]/2)]:
                    if min(x,y,1023-x,1023-y,x-x0,y-y0,x0+127-x,y0+127-y)<r+1:
                        valid=False;break
                if not valid:break
                observations.append(dict(frame=frame,delta=d,track_id=track_id,truth_particle_id=particle_id,
                    x=float(p[1]),y=float(p[2]),z=float(p[3]),diameter_px=float(p[4]),
                    truth_x=float(q.x_px),truth_y=float(q.y_px),truth_diameter_um=float(q.diameter_um),
                    one_to_one_mapping_present=mapped is not None,nearest_truth_particle_is_expected=True,
                    xy_error_px=xy,xy_error_um=xy*10,relative_radius_error=xy/radius,
                    z_error_um=float((p[3]-1)*100-q.z_um)))
            if not valid:break
        if not valid:continue
        counts['all_ten_nearest_expected_truth_and_whole_disks_visible']+=1
        if pre_gap<1:continue
        if max(z_gaps)>500:continue
        counts['pre_separation_and_common_focus']+=1
        item={**row,'truth_event':{'collision_event_id':int(e.name),**e.to_dict()},'observations':observations,
            'max_xy_error_px':max(p['xy_error_px'] for p in observations),
            'mean_xy_error_px':float(np.mean([p['xy_error_px'] for p in observations])),
            'max_relative_radius_error':max(p['relative_radius_error'] for p in observations),
            'min_parent_diameter_um':float(min(e.parent1_diameter_um,e.parent2_diameter_um)),
            'pre_edge_gap_px':float(pre_gap),'max_pre_z_separation_um':float(max(z_gaps)),
            'max_abs_z_error_um':max(abs(p['z_error_um']) for p in observations),
            'final_candidate_retained':True,'display_deltas':list(DELTAS),
            'display_reference_frame':t,'scoring_candidate_offset_from_truth_frames':score_t-t}
        eligible.append(item)
    for threshold in [1,1.5,2,3,4]:
        counts[f'max_xy_error_le_{threshold}_px']=sum(r['max_xy_error_px']<=threshold for r in eligible)
    precise=[r for r in eligible if r['max_xy_error_px']<=3 and r['max_relative_radius_error']<=.35 and r['max_abs_z_error_um']<=250]
    large=[r for r in eligible if r['min_parent_diameter_um']>=180 and r['max_xy_error_px']<=5 and r['max_relative_radius_error']<=.35 and r['max_abs_z_error_um']<=250]
    for r in precise:r['display_group']='position_priority'
    for r in large:r['display_group']='larger_particles'
    preview=sorted(large,key=lambda r:-r['min_parent_diameter_um'])+sorted(precise,key=lambda r:r['max_xy_error_px'])
    for i,r in enumerate(preview,1):r['review_id']=f'P{i:02d}'
    write('eligible.json',sorted(eligible,key=lambda r:r['max_xy_error_px']))
    write('shortlist.json',preview)
    write('search-protocol.json',dict(counts=counts,selected_for_preview=len(preview),
        constraints='Strict identity consistent at the scoring-candidate time; final retained; both scores >=0.75 (the author\'s >=0.9 constraint applies to the artifact, not this true-event search); display time zero is exact truth-event time (scoring candidate offset recorded separately); parents >=150um; all 10 displayed observations are nearest to expected truth parents/daughter with no conflicting one-to-one identity (missing assignments audited geometrically); position-priority group max XY error<=3px; larger-particle group both parents>=180um and max XY error<=5px; both groups <=35% truth radius and abs Z error<=250um; pre-track Z separation<=500um; separated pre-event disks; no missing/interpolated markers and no terminating observations from event time.',
        search_note='Initial <=2px limit yielded no examples meeting all six-frame/size/identity conditions; tested and recorded thresholds through 4px, then retained two explicit illustration tiers. These criteria do not change any performance evaluation.',
        note='Post-hoc illustration selection only; truth used for auditing and selection, never to move displayed points or refocus images.'))
    print(json.dumps(counts,indent=2))
    for r in preview[:12]:
        e=r['truth_event'];print(r['review_id'],int(r['candidate_frame']),int(r['vanished_track_id']),int(r['survivor_track_id']),
            'D',round(e['parent1_diameter_um']),round(e['parent2_diameter_um']),round(e['child_diameter_um']),
            'XYmax',round(r['max_xy_error_px'],3),'scores',round(r['static_score'],3),round(r['base_score'],3))


if __name__=='__main__':main()
