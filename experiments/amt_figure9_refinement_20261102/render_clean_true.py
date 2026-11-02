#!/usr/bin/env python3
"""Render alternatives without altering measurements, smoothing, or the paper figure."""
import json
from clean_true import OUT, write, REPO, ROOT
import frame_gallery as gallery
import frame_gallery_tracks as overlay
from search import original
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d._core.track_images import point_at


def main():
    eligible=json.loads((OUT/'eligible.json').read_text())
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    torch.set_num_threads(4)
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cpu')
    reviews=[]
    for number,score_frame in enumerate([4335,3593,5249,4740],1):
        r=next(r for r in eligible if r['candidate_frame']==score_frame)
        case_id=f'C{number}';folder=OUT/case_id
        for sub in ['images','tracks','cache']:(folder/sub).mkdir(parents=True,exist_ok=True)
        gallery.OUT=overlay.OUT=folder
        v=tracks.loc[tracks.track_id.eq(r['vanished_track_id'])].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(r['survivor_track_id'])].sort_values('frame')
        panels=[];audits=[]
        for i,d in enumerate(range(-4,2),1):
            f=r['display_reference_frame']+d;point=point_at(s,f,nearest=False)
            assert point is not None
            with torch.no_grad():plane=recon.intensity(f,point.z)
            raw,x0,y0,valid=original.crop_with_black_padding(plane,point.x,point.y,128)
            display=original.normalize_valid_crop(raw,valid)
            name=f'{case_id}-{i}-frame-{f}.png';Image.fromarray(display).save(folder/'images'/name)
            np.savez_compressed(folder/'cache'/f'{i}.npz',raw=raw,valid=valid,display=display)
            panel=dict(label=f'{case_id}-{i}',filename=name,frame=f,delta=d,time_ms=d/4,x0=x0,y0=y0,
                center_x=point.x,center_y=point.y,center_z=point.z,
                recorded_fraction=float(valid.mean()),png_sha256=gallery.sha(folder/'images'/name))
            panels.append(panel);audits.append(overlay.render_overlay(panel,v,s))
        e=r['truth_event']
        title=f"{case_id}: {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} -> {e['child_diameter_um']:.0f} um"
        gallery.contact_sheet(title,panels,OUT/f'{case_id}-frames.png',with_tracks=True)
        reviews.append(dict(case_id=case_id,source=r,panels=panels,overlay_audit=audits))
        print(title,'XY max',r['max_xy_error_px'],'mean',r['mean_xy_error_px'],flush=True)
    write('rendered-cases.json',reviews)


if __name__=='__main__':main()
