#!/usr/bin/env python3
"""Inspect high-score artifacts using the original target-centered display policy."""
import json
import pickle
from pathlib import Path
import search as base
from search import REPO, ROOT, original
import numpy as np
import pandas as pd
import torch
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d._core.track_images import point_at
import matplotlib.pyplot as plt

OUT=REPO.parent/'audit/revision-figure9-centered-20261102'
OUT.mkdir(exist_ok=True)
SPECS=[
    ('C01',12486,12575,3859,1165,(-4,0,1)),
    ('C02',4342,4349,1767,348,(-4,0,1)),
    ('C03',9260,9259,3009,833,(-4,0,1)),
    ('C04',18638,18667,5523,1815,(-4,0,1)),
    ('H01',8518,8507,2813,None,(-4,0,1)),
    ('H02',15431,15514,4761,None,(-3,0,4)),
]


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def main():
    cached=REPO.parent/'audit/revision-figure9-larger-20261102/pool.pkl'
    if cached.exists():
        with cached.open('rb') as f:
            pool,truth,tracks,lookup,arrays,keys=pickle.load(f)
    else:
        pool,truth,tracks,lookup,arrays,keys=base.load_pool()
    torch.set_num_threads(4)
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cpu')
    (OUT/'cache').mkdir(exist_ok=True)
    records=[];keep=set()
    for ident,v,s,t,gt,deltas in SPECS:
        r=pool.loc[pool.vanished_track_id.eq(v)&pool.survivor_track_id.eq(s)&pool.candidate_frame.eq(t)]
        assert len(r)==1
        row=r.iloc[0].to_dict();row.update(review_id=ident,display_deltas=list(deltas),
            kind='true_collision' if gt is not None else 'shared_particle_artifact',
            final_candidate_retained=(v,s,t) in keys)
        if gt is not None:
            row['truth_event']=truth.loc[truth.collision_event_id.eq(gt)].iloc[0].to_dict()
        if ident=='H01':
            row.update(source_particle_id=3605,identity_evidence='Independent projected-disk and axial-distance audit; the one-to-one mapping assigns this particle to the continuing ID only.')
        elif ident=='H02':
            row['kind']='crossing_track_artifact'
            row['identity_evidence']='Distinct truth particles 6319 and 6377 cross in projection at different depths; do not label this a shared-particle example.'
        if gt is None:
            nearby=truth.loc[(truth.frame-t).abs().le(5)]
            assert not np.hypot(nearby.x_um-row['x_um'],nearby.y_um-row['y_um']).le(350).any()
            assert row['static_score']>=.9 and row['static_score']-row['base_score']>=.3
        vt=tracks.loc[tracks.track_id.eq(v)].sort_values('frame');st=tracks.loc[tracks.track_id.eq(s)].sort_values('frame')
        keep.update([v,s]);images={};panels=[]
        for d in deltas:
            frame=t+d;sp=point_at(st,frame,nearest=False);vp=point_at(vt,frame,nearest=False)
            assert sp is not None,(ident,frame)
            with torch.no_grad():plane=recon.intensity(frame,sp.z)
            raw,x0,y0,valid=original.crop_with_black_padding(plane,sp.x,sp.y,128)
            display=original.normalize_valid_crop(raw,valid)
            images[f'raw_{d}']=raw;images[f'valid_{d}']=valid;images[f'display_{d}']=display
            wide,_,_,wv=original.crop_with_black_padding(plane,sp.x,sp.y,256)
            images[f'wide_{d}']=wide;images[f'wide_valid_{d}']=wv
            panels.append(dict(frame=frame,delta=d,center_x=sp.x,center_y=sp.y,center_z=sp.z,
                x0=x0,y0=y0,recorded_fraction=float(valid.mean()),
                contrast_low_high=np.percentile(raw[valid],[1,99]).tolist(),
                actual_marker_observations={'terminating':vp is not None,'continuing':True}))
        np.savez_compressed(OUT/f'cache/{ident}.npz',**images)
        dump(f'cache/{ident}.json',{'review_id':ident,'panels':panels})
        row['pre_diameter_min_um']=min(point_at(vt,t+deltas[0]).diameter_px,point_at(st,t+deltas[0]).diameter_px)*10
        records.append(row)
        fig,axes=plt.subplots(1,3,figsize=(10,3.9))
        for ax,p in zip(axes,panels):
            ax.imshow(images[f"display_{p['delta']}"],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for track,color,marker in [(vt,original.TERMINATING_COLOR,'o'),(st,original.CONTINUING_COLOR,'s')]:
                point=point_at(track,p['frame'],nearest=False)
                if point is not None:
                    original.overlay_track_tail(ax,track,p['frame'],p['x0'],p['y0'],color)
                    original.overlay_track_marker(ax,point,p['x0'],p['y0'],color=color,marker=marker)
            ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f"{ident}: {p['delta']/4:+.2f} ms; recorded {p['recorded_fraction']:.0%}",fontsize=11)
        title=f"{ident}: {row['static_score']:.3f} → {row['base_score']:.3f}"
        if gt is not None:
            e=row['truth_event'];title+=f"; D {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} → {e['child_diameter_um']:.0f} µm"
        else:title+=f"; {row['kind']}"
        fig.suptitle(title,fontsize=12);fig.tight_layout();fig.savefig(OUT/f'{ident}.png',dpi=165);plt.close(fig)
        print(title,flush=True)
    dump('shortlist.json',records)
    tracks.loc[tracks.track_id.isin(keep)].to_parquet(OUT/'shortlist-tracks.parquet',index=False)


if __name__=='__main__':main()
