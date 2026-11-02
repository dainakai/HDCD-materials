#!/usr/bin/env python3
"""Find visible illustrative cases using frozen results; reconstruct only display crops."""
from pathlib import Path
import sys, json, hashlib, importlib.util
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'experiments/amt_full_20261101'))
from common import ROOT, path, tracking, events
import numpy as np
import pandas as pd
import torch
from ra3d.strict_labels import attach_identity_labels
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d.config import KEY_COLUMNS
from ra3d.tracks import read_tracks
from ra3d._core.track_images import point_at
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=REPO.parent/'audit/revision-figure9-refinement-20261102'
OUT.mkdir(parents=True,exist_ok=True)
DELTAS=(-5,0,4)
spec=importlib.util.spec_from_file_location('original_plot',HERE/'source/original_plot.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def load_pool():
    torch.set_num_threads(4)
    full=pd.read_parquet(path(260745,'ours','paper92.parquet'))
    full=full.merge(pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet'),on=KEY_COLUMNS,validate='one_to_one')
    mapping=pd.read_parquet(tracking(260745,'ours')/'mapping.parquet')
    truth=events(260745)
    tracks=read_tracks(tracking(260745,'ours')/'smoothed.parquet')
    selected=full.loc[full.image_enriched.eq(1)&full.rendered_pre_frames.ge(7)].copy()
    # Label against the full scoring union, as in the finalized illustration audit.
    full,_,_=attach_identity_labels(full,truth,mapping,z_tolerance_um=15000.)
    selected=full.loc[full.image_enriched.eq(1)&full.rendered_pre_frames.ge(7)].copy()
    lookup=mapping.set_index(['frame','observed_track_id']).source_particle_id
    arrays={int(k):v.sort_values('frame')[['frame','x','y','z','diameter_px']].to_numpy(float) for k,v in tracks.groupby('track_id',sort=False)}
    pred=pd.read_parquet(ROOT/'results/ours/predictions.parquet')
    keys=set(zip(pred.vanished_track_id,pred.survivor_track_id,pred.candidate_frame_raw))
    return selected,truth,tracks,lookup,arrays,keys

def geometry(row,arrays,side=128,artifact=False):
    t=int(row.candidate_frame);v=arrays[int(row.vanished_track_id)];s=arrays[int(row.survivor_track_id)]
    if not all(f in s[:,0] for f in [t+d for d in DELTAS]):return None
    if t+DELTAS[0] not in v[:,0] or t+4 in v[:,0]:return None
    if artifact and (t not in v[:,0] or v[-1,0]>t+1):return None
    for f in [t+d for d in DELTAS]:
        sx,sy=[np.interp(f,s[:,0],s[:,i]) for i in [1,2]]
        if not(side/2+4<=sx<=1024-side/2-4 and side/2+4<=sy<=1024-side/2-4):return None
        if f in v[:,0]:
            vx,vy,vd=[np.interp(f,v[:,0],v[:,i]) for i in [1,2,4]]
            if max(abs(vx-sx),abs(vy-sy))+vd/2>side/2-5:return None
    f=t+DELTAS[0]
    vp=np.array([np.interp(f,v[:,0],v[:,i]) for i in range(1,5)])
    sp=np.array([np.interp(f,s[:,0],s[:,i]) for i in range(1,5)])
    return {'pre_xy_separation_px':float(np.linalg.norm(vp[:2]-sp[:2])),
            'pre_z_separation_um':float(abs(vp[2]-sp[2])*100),
            'pre_diameter_min_um':float(min(vp[3],sp[3])*10),
            'pre_edge_gap_px':float(np.linalg.norm(vp[:2]-sp[:2])-(vp[3]+sp[3])/2)}

def select():
    selected,truth,tracks,lookup,arrays,keys=load_pool()
    truth_by_id=truth.set_index('collision_event_id')
    true_rows=[]
    for _,r in selected.loc[selected.identity_state.eq('identity_consistent')&selected.static_score.ge(.9)&selected.base_score.ge(.9)].iterrows():
        e=truth_by_id.loc[int(r.matched_truth_event_id)]
        if int(r.candidate_frame)!=int(e.frame):continue
        if (r.vanished_track_id,r.survivor_track_id,r.candidate_frame) not in keys:continue
        g=geometry(r,arrays)
        if g is None or g['pre_edge_gap_px']<2:continue
        item=r.to_dict();item.update(g)
        item.update(kind='true_collision',truth_event=e.to_dict(),final_candidate_retained=True)
        item['visibility_priority']=float(min(e.parent1_diameter_um,e.parent2_diameter_um))-.05*g['pre_z_separation_um']
        true_rows.append(item)
    true=pd.DataFrame(true_rows).sort_values(['visibility_priority','static_score'],ascending=False).drop_duplicates('matched_truth_event_id').head(30)
    artifact_rows=[]
    for _,r in selected.loc[selected.spatial_positive.eq(0)&selected.pre_same_interp_fraction.ge(7/9)&selected.static_score.ge(.35)&(selected.static_score-selected.base_score).ge(.1)].iterrows():
        if (r.vanished_track_id,r.survivor_track_id,r.candidate_frame) in keys:continue
        nearby=truth.loc[(truth.frame-r.candidate_frame).abs().le(5)]
        if np.hypot(nearby.x_um-r.x_um,nearby.y_um-r.y_um).le(350).any():continue
        g=geometry(r,arrays,artifact=True)
        if g is None:continue
        t=int(r.candidate_frame);v=int(r.vanished_track_id);s=int(r.survivor_track_id)
        entries=[[lookup.get((f,track),np.nan) for track in [v,s]] for f in range(t-10,t-1)]
        values=[int(x) for pair in entries for x in pair if np.isfinite(x)]
        if not values or len(set(values))!=1:continue
        source=values[0]
        support=[sum(pair[i]==source for pair in entries) for i in range(2)]
        distinct=sum(source in pair for pair in entries)
        if min(support)<2 or distinct<5:continue
        item=r.to_dict();item.update(g)
        item.update(kind='shared_particle_artifact',true_event_id=-1,source_particle_id=source,identity_support=support,
                    identity_distinct_frames=distinct,final_candidate_retained=False,
                    score_decrease=float(r.static_score-r.base_score))
        artifact_rows.append(item)
    artifact=pd.DataFrame(artifact_rows)
    if len(artifact):
        artifact=artifact.sort_values(['score_decrease','pre_diameter_min_um'],ascending=False).drop_duplicates(['source_particle_id']).head(18)
    records=[]
    for prefix,df in [('T',true),('A',artifact)]:
        for i,row in enumerate(df.to_dict('records')):
            row['review_id']=f'{prefix}{i+1:02d}';records.append(row)
    dump('shortlist.json',records)
    keep=set(int(r[c]) for r in records for c in ['vanished_track_id','survivor_track_id'])
    tracks.loc[tracks.track_id.isin(keep)].to_parquet(OUT/'shortlist-tracks.parquet',index=False)
    dump('search-protocol.json',{'true_candidates_eligible':len(true_rows),'artifact_candidates_eligible':len(artifact_rows),
         'display_candidates':len(records),'crop_size_px':128,'contrast':'Per-frame 1st/99th percentiles, exact original function.',
         'time_frames':list(DELTAS),'focus':'Continuing-track smoothed Z, no manual focus or truth-based refocusing.',
         'selection':'Illustrative visual selection at author request after numerical evaluation; no fitting, threshold or performance metric changes.',
         'truth_conditions':'Identity-consistent, exact event frame, final policy retained, both scores >=0.9, both pre-event observations and continuing observations at all display frames present, no terminating observation in the post-frame, both pre-event disks in crop with >=2 px edge separation.',
         'artifact_conditions':'No ground truth in the common evaluation gate (time +/-5, XY 350 um, no Z condition); both track IDs supported by same truth particle over nine preframes; >=5 distinct support frames and >=2 per track; no conflicting IDs; absent from final policy.',
         'style_source_sha256':hashlib.sha256((HERE/'source/original_plot.py').read_bytes()).hexdigest()})
    print(json.dumps({'true':len(true),'artifact':len(artifact),'eligible_true':len(true_rows),'eligible_artifact':len(artifact_rows)}),flush=True)

def render():
    torch.set_num_threads(4)
    records=json.loads((OUT/'shortlist.json').read_text())
    tracks=pd.read_parquet(OUT/'shortlist-tracks.parquet')
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cpu')
    (OUT/'cache').mkdir(exist_ok=True)
    metadata=[]
    for row in records:
        ident=row['review_id'];cache=OUT/f'cache/{ident}.npz'
        if cache.exists():
            metadata.append(json.loads((OUT/f'cache/{ident}.json').read_text()));continue
        t=int(row['candidate_frame']);sv=tracks.loc[tracks.track_id.eq(row['survivor_track_id'])].sort_values('frame')
        panels=[];images={}
        for delta in DELTAS:
            f=t+delta
            sx,sy,sz=[float(np.interp(f,sv.frame,sv[col])) for col in ['x','y','z']]
            with torch.no_grad():plane=recon.intensity(f,sz)
            crop,x0,y0,valid=original.crop_with_black_padding(plane,sx,sy,128)
            assert valid.all()
            normalized=original.normalize_valid_crop(crop,valid)
            images[f'raw_{delta}']=crop;images[f'display_{delta}']=normalized
            # Retain a wider crop solely for the requested field-of-view comparison.
            wide,_,_,valid_wide=original.crop_with_black_padding(plane,sx,sy,256)
            images[f'wide_{delta}']=wide;images[f'wide_valid_{delta}']=valid_wide
            panels.append({'frame':f,'delta':delta,'x0':x0,'y0':y0,'center_x':sx,'center_y':sy,'center_z':sz,
                           'recorded_fraction':float(valid.mean()),'contrast_low_high':np.percentile(crop,[1,99]).tolist()})
        np.savez_compressed(cache,**images)
        meta={'review_id':ident,'panels':panels};dump(f'cache/{ident}.json',meta);metadata.append(meta)
        print(f"Rendered {ident}: {row['static_score']:.3f} -> {row['base_score']:.3f}",flush=True)
    for kind in ['true_collision','shared_particle_artifact']:
        group=[r for r in records if r['kind']==kind]
        for batch in range(0,len(group),6):
            subset=group[batch:batch+6]
            fig,axes=plt.subplots(len(subset),3,figsize=(9,2.6*len(subset)),squeeze=False)
            for ir,row in enumerate(subset):
                ident=row['review_id'];cache=np.load(OUT/f'cache/{ident}.npz');meta=json.loads((OUT/f'cache/{ident}.json').read_text())
                v=tracks.loc[tracks.track_id.eq(row['vanished_track_id'])];s=tracks.loc[tracks.track_id.eq(row['survivor_track_id'])]
                for col,p in enumerate(meta['panels']):
                    ax=axes[ir,col];ax.imshow(cache[f"display_{p['delta']}"],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    for tr,color,marker in [(v,original.TERMINATING_COLOR,'o'),(s,original.CONTINUING_COLOR,'s')]:
                        point=point_at(tr,p['frame'],nearest=False)
                        if point is not None:
                            original.overlay_track_tail(ax,tr,p['frame'],p['x0'],p['y0'],color)
                            original.overlay_track_marker(ax,point,p['x0'],p['y0'],color=color,marker=marker)
                    ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title(f"{ident}: {p['delta']/4:+.1f} ms",fontsize=10)
                detail=f"{row['static_score']:.3f} -> {row['base_score']:.3f}; frame {int(row['candidate_frame'])}"
                if kind=='true_collision':
                    e=row['truth_event'];detail+=f"; D={e['parent1_diameter_um']:.0f}+{e['parent2_diameter_um']:.0f}"
                axes[ir,1].set_xlabel(detail,fontsize=9)
            fig.tight_layout();fig.savefig(OUT/f'{kind}-sheet-{batch//6+1}.png',dpi=140);plt.close(fig)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='render':render()
    else:select()
