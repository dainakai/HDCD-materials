#!/usr/bin/env python3
"""Find larger particles in frozen candidates, using wholly recorded edge crops."""
import json
import pickle
import sys
from pathlib import Path
import search as base
from search import REPO, ROOT, original, DELTAS, load_pool
import numpy as np
import pandas as pd
import torch
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d._core.track_images import point_at
import matplotlib.pyplot as plt

OUT = REPO.parent/'audit/revision-figure9-larger-20261102'
OUT.mkdir(exist_ok=True)


def dump(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def geometry(r, arrays, artifact=False):
    t=int(r.candidate_frame);v=arrays[int(r.vanished_track_id)];s=arrays[int(r.survivor_track_id)]
    if not all(t+d in s[:,0] for d in DELTAS):return None
    if t-5 not in v[:,0] or t+4 in v[:,0]:return None
    if artifact and (t not in v[:,0] or v[-1,0]>t+1):return None
    panels=[]
    for d in DELTAS:
        frame=t+d;sp=s[s[:,0]==frame][0];vp=v[v[:,0]==frame]
        # Translate the entire 128-pixel window into the recorded frame.
        # No pixels are synthesized, reflected, shifted within a crop or rescaled.
        cx,cy=np.clip(sp[1:3],64,960)
        x0,y0=int(np.rint(cx))-64,int(np.rint(cy))-64
        for p in [sp,*vp]:
            margin=min(p[1]-x0, x0+127-p[1], p[2]-y0, y0+127-p[2])-p[4]/2
            if margin<1:return None
        panels.append(dict(frame=frame,delta=d,center_x=float(sp[1]),center_y=float(sp[2]),center_z=float(sp[3]),
                           crop_center_x=float(cx),crop_center_y=float(cy)))
    pre_v=v[v[:,0]==t-5][0];pre_s=s[s[:,0]==t-5][0]
    sep=np.linalg.norm(pre_v[1:3]-pre_s[1:3])
    return dict(panels=panels,pre_diameter_min_um=float(min(pre_v[4],pre_s[4])*10),
                pre_edge_gap_px=float(sep-(pre_v[4]+pre_s[4])/2),pre_z_separation_um=float(abs(pre_v[3]-pre_s[3])*100))


def select():
    cache=OUT/'pool.pkl'
    if cache.exists():
        with cache.open('rb') as f:selected,truth,tracks,lookup,arrays,keys=pickle.load(f)
    else:
        selected,truth,tracks,lookup,arrays,keys=load_pool()
    truth_by_id=truth.set_index('collision_event_id')
    true=[];artifact=[]
    for _,r in selected.loc[selected.identity_state.eq('identity_consistent')&selected.static_score.ge(.75)&selected.base_score.ge(.75)].iterrows():
        e=truth_by_id.loc[int(r.matched_truth_event_id)]
        if int(r.candidate_frame)!=int(e.frame):continue
        if (r.vanished_track_id,r.survivor_track_id,r.candidate_frame) not in keys:continue
        if min(e.parent1_diameter_um,e.parent2_diameter_um)<165:continue
        g=geometry(r,arrays)
        if g is None or g['pre_edge_gap_px']<1:continue
        item=r.to_dict();item.update(g)
        item.update(kind='true_collision',truth_event=e.to_dict(),final_candidate_retained=True,
                    size_priority=float(min(e.parent1_diameter_um,e.parent2_diameter_um)))
        true.append(item)
    for _,r in selected.loc[selected.spatial_positive.eq(0)&selected.pre_same_interp_fraction.ge(7/9)&selected.static_score.ge(.35)&(selected.static_score-selected.base_score).ge(.1)].iterrows():
        t=int(r.candidate_frame);v=int(r.vanished_track_id);s=int(r.survivor_track_id)
        if (v,s,t) in keys:continue
        nearby=truth.loc[(truth.frame-t).abs().le(5)]
        if np.hypot(nearby.x_um-r.x_um,nearby.y_um-r.y_um).le(350).any():continue
        g=geometry(r,arrays,artifact=True)
        if g is None or g['pre_diameter_min_um']<170:continue
        entries=[[lookup.get((f,track),np.nan) for track in [v,s]] for f in range(t-10,t-1)]
        vals=[int(x) for pair in entries for x in pair if np.isfinite(x)]
        if not vals or len(set(vals))!=1:continue
        source=vals[0];support=[sum(pair[i]==source for pair in entries) for i in range(2)]
        distinct=sum(source in pair for pair in entries)
        if min(support)<2 or distinct<5:continue
        item=r.to_dict();item.update(g)
        item.update(kind='shared_particle_artifact',true_event_id=-1,source_particle_id=source,
                    identity_support=support,identity_distinct_frames=distinct,final_candidate_retained=False,
                    score_decrease=float(r.static_score-r.base_score))
        artifact.append(item)
    true=sorted(true,key=lambda r:r['size_priority'],reverse=True)
    artifact=sorted(artifact,key=lambda r:r['pre_diameter_min_um'],reverse=True)
    rows=[]
    for prefix,candidates in [('L',true),('B',artifact)]:
        seen=set()
        for r in candidates:
            key=r['matched_truth_event_id'] if prefix=='L' else r['source_particle_id']
            if key in seen:continue
            seen.add(key)
            r['review_id']=f'{prefix}{len(seen):02d}';rows.append(r)
            if len(seen)==12:break
    dump('shortlist.json',rows)
    keep={int(r[k]) for r in rows for k in ['vanished_track_id','survivor_track_id']}
    tracks.loc[tracks.track_id.isin(keep)].to_parquet(OUT/'shortlist-tracks.parquet',index=False)
    dump('search-protocol.json',dict(true_candidates_eligible=len(true),artifact_candidates_eligible=len(artifact),
         display_candidates=len(rows),crop_size_px=128,time_frames=list(DELTAS),
         contrast='Same original per-frame first/99th-percentile linear normalization.',
         crop_position='Continuing-track center, clamped to [64,960] in each direction to retain a wholly recorded 128-pixel crop. Actual measured disks have >=1 pixel clearance from every crop boundary.',
         focus='Continuing-track smoothed Z; no manual or truth-based refocusing.',
         selection='Author requested larger particles. Post-hoc illustrative choice, sorted by true smaller-parent diameter / measured artifact diameter, followed by visual inspection. No change to models, inference or performance.',
         truth_conditions='Both true parent diameters >=165um; exact truth-event frame; identity-consistent and final-policy retained; saved scores >=0.75; actual observations; measured pre-event edge gap >=1px.',
         artifact_conditions='Measured smaller-track diameter >=170um; no truth within common +/-5frame/350umXY/noZ gate; no conflicting truth identity over nine preframes; >=5 distinct supported frames and >=2 per track; shared component >=7/9; saved score decrease >=0.1; final-policy absent.'))
    for r in rows:
        e=r.get('truth_event',{});print(r['review_id'],int(r['candidate_frame']),e.get('parent1_diameter_um'),e.get('parent2_diameter_um'),r['pre_diameter_min_um'],r['static_score'],r['base_score'])


def render():
    torch.set_num_threads(4)
    rows=json.loads((OUT/'shortlist.json').read_text());tracks=pd.read_parquet(OUT/'shortlist-tracks.parquet')
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cpu')
    (OUT/'cache').mkdir(exist_ok=True)
    for r in rows:
        ident=r['review_id'];dest=OUT/f'cache/{ident}.npz'
        if dest.exists():
            with np.load(dest) as cached:
                if all(cached[f'wide_valid_{d}'].all() for d in DELTAS):continue
        images={};panels=[]
        for p in r['panels']:
            with torch.no_grad():plane=recon.intensity(p['frame'],p['center_z'])
            raw,x0,y0,valid=original.crop_with_black_padding(plane,p['crop_center_x'],p['crop_center_y'],128)
            assert valid.all()
            d=p['delta'];images[f'raw_{d}']=raw;images[f'display_{d}']=original.normalize_valid_crop(raw,valid)
            wide_cx,wide_cy=np.clip([p['center_x'],p['center_y']],128,896)
            wide,_,_,wv=original.crop_with_black_padding(plane,wide_cx,wide_cy,256)
            assert wv.all()
            images[f'wide_{d}']=wide;images[f'wide_valid_{d}']=wv
            panels.append({**p,'x0':x0,'y0':y0,'recorded_fraction':1.,'contrast_low_high':np.percentile(raw,[1,99]).tolist()})
        np.savez_compressed(dest,**images);dump(f'cache/{ident}.json',{'review_id':ident,'panels':panels})
        print('Rendered',ident,flush=True)
    for kind in ['true_collision','shared_particle_artifact']:
        group=[r for r in rows if r['kind']==kind]
        for start in range(0,len(group),3):
            subset=group[start:start+3];fig,axes=plt.subplots(len(subset),3,figsize=(10,3.1*len(subset)),squeeze=False)
            for ri,r in enumerate(subset):
                ident=r['review_id'];images=np.load(OUT/f'cache/{ident}.npz');meta=json.loads((OUT/f'cache/{ident}.json').read_text())
                for ci,p in enumerate(meta['panels']):
                    ax=axes[ri,ci];ax.imshow(images[f"display_{p['delta']}"],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    for track,color,marker in [(r['vanished_track_id'],original.TERMINATING_COLOR,'o'),(r['survivor_track_id'],original.CONTINUING_COLOR,'s')]:
                        tr=tracks.loc[tracks.track_id.eq(track)].sort_values('frame');point=point_at(tr,p['frame'],nearest=False)
                        if point is not None:
                            original.overlay_track_tail(ax,tr,p['frame'],p['x0'],p['y0'],color)
                            original.overlay_track_marker(ax,point,p['x0'],p['y0'],color=color,marker=marker)
                    ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title(f"{ident}: {p['delta']/4:+.2f} ms",fontsize=11)
                note=f"score {r['static_score']:.3f} → {r['base_score']:.3f}"
                if kind=='true_collision':
                    e=r['truth_event'];note+=f"; D {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} → {e['child_diameter_um']:.0f} µm"
                else:note+=f"; measured D {r['pre_diameter_min_um']:.0f} µm"
                axes[ri,1].set_xlabel(note,fontsize=11)
            fig.tight_layout();fig.savefig(OUT/f'{kind}-sheet-{start//3+1}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='render':render()
    else:select()
