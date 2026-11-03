"""Reconstruct ten illustrative frames using the campaign's frozen phase-3 engine."""
import sys
import os
from common import *
ENGINE = WORK/'audit/revision-full-20261101/src'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ENGINE))
sys.path.append(str(REPO/'.venv/lib/python3.12/site-packages'))
for name in ['MPLCONFIGDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TMPDIR']:
    path=ROOT/'cache'/name.lower();path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
import numpy as np
import pandas as pd
import torch
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d._core.track_images import EventSpec,choose_center,crop_with_mean_padding,normalize_crop

CASES=[('a','21:2906:3007:3612',3608),('b','17:10138:9881:8565',8563)]

def main(cases=CASES, folder='case_images', offsets=(-4,-2,0,2,4),
         origin_policy='Illustrative contact/merging frame chosen from reviewed movie, not a change to candidate time or analysis windows.',
         crop_center_policy='review_focus'):
    for name,expected in read(REPO/'experiments/amt_full_20261101/engine-manifest.json')['files'].items():
        assert sha(ENGINE/name)==expected,name
    torch.set_num_threads(4)
    output=PRIVATE/folder;output.mkdir(exist_ok=True)
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet')
    records=[]
    for tag,cid,display_frame in cases:
        event=labels.loc[labels.candidate_id.eq(cid)].iloc[0]
        assert event.status=='collision'
        scene=int(event.scene_num);root=CAMPAIGN/f'scenes/scene{scene:04d}'
        manifest=read(root/'review/manifest.json')
        detail=next(x for x in manifest['candidates'] if x['candidate_id']==cid)
        assert sha(CAMPAIGN/detail['asset'])==event.asset_sha256
        tracks=pd.read_parquet(root/'tracking/smoothed.parquet')
        a=tracks.loc[tracks.track_id.eq(event.vanished_track_id)].reset_index(drop=True)
        b=tracks.loc[tracks.track_id.eq(event.survivor_track_id)].reset_index(drop=True)
        spec=EventSpec('figure10',0,'',int(event.candidate_frame_raw),int(event.vanished_track_id),int(event.survivor_track_id),
            float(event.vanished_diameter_um),float(event.survivor_diameter_um),float(event.score),'survivor_remains')
        acquisition=Acquisition.load(root/'input/acquisition.yaml')
        assert acquisition.phase_iterations==3 and acquisition.padding_px==2048 and acquisition.reconstruction_start_um==74900
        recon=Reconstructor(acquisition,'cuda:0')
        frames=[]
        for offset in offsets:
            frame=display_frame+offset
            center=choose_center(spec,a,b,frame)
            intensity=recon.intensity(frame,center.z)
            cx,cy=center.x,center.y
            if crop_center_policy=='observed_parent_bounds':
                observed=pd.concat([a.loc[a.frame.eq(frame)],b.loc[b.frame.eq(frame)]])
                if len(observed):
                    radius=observed.diameter_px/2
                    lower=np.array([(observed.x-radius).min(),(observed.y-radius).min()])
                    upper=np.array([(observed.x+radius).max(),(observed.y+radius).max()])
                    assert np.max(upper-lower)<124,(cid,frame,'parent extents exceed the display crop')
                    cx,cy=(lower+upper)/2
            crop,x0,y0=crop_with_mean_padding(intensity,cx,cy,128)
            path=output/f'{tag}_{frame}.npz'
            np.savez_compressed(path,intensity=crop,display=normalize_crop(crop))
            frames.append(dict(frame=frame,offset_ms=offset/4,x0=x0,y0=y0,z_slice=center.z,
                cache=path.name,sha256=sha(path),pixel_pitch_um=10,scale_bar_um=500,scale_bar_px=50))
        record=dict(case=tag,candidate_id=cid,scene_num=scene,review_rank=int(event.review_rank),candidate_frame_raw=int(event.candidate_frame_raw),
            display_t0_frame=display_frame,display_t0_policy=origin_policy,
            parent_diameters_um=[float(event.pre_diameter_vanished_um),float(event.pre_diameter_survivor_um)],post_diameter_um=float(event.post_diameter_um),
            frames=frames,device=torch.cuda.get_device_name(0),phase_iterations=3,padding_px=2048,
            acquisition_sha256=sha(root/'input/acquisition.yaml'),review_asset_sha256=event.asset_sha256)
        record['crop_center_policy']=crop_center_policy
        records.append(record)
        print(f'Reconstructed case {tag}: {cid}',flush=True)
    save(output/'manifest.json',{'created_at':stamp(),'cases':records,'engine_manifest_sha256':sha(REPO/'experiments/amt_full_20261101/engine-manifest.json')})

if __name__=='__main__':main()
