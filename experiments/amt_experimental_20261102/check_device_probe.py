"""Bounded cross-device check against the completed 128-frame GPU0 smoke run."""
from common import *
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def compare(label):
    baseline=ROOT/'smoke/128/scene0012'
    other=ROOT/f'distributed/probes/{label}/scene0012'
    stages={p.stem:read(p) for p in (other/'stages').glob('*.json')}
    assert set(stages)=={'background','minip','measurement','tracking','inference','assets'}
    images={}
    for kind,relative in [('background_C001','prepared/01_bkgrem/C001H001S0012'),
                          ('background_C002','prepared/01_bkgrem/C002H001S0012'),('minip','minip')]:
        names=sorted(p.name for p in (baseline/relative).glob('*.png'))
        assert names==sorted(p.name for p in (other/relative).glob('*.png')) and len(names)==128
        equal=0;maximum=0;changed=0;pixels=0
        for name in names:
            a=np.asarray(Image.open(baseline/relative/name)).astype(np.int16)
            b=np.asarray(Image.open(other/relative/name)).astype(np.int16)
            assert a.shape==b.shape==(1024,1024)
            delta=np.abs(a-b);maximum=max(maximum,int(delta.max()));changed+=int(np.count_nonzero(delta));pixels+=a.size
            equal+=int(sha(baseline/relative/name)==sha(other/relative/name))
        images[kind]={'bit_identical_files':equal,'files':128,'maximum_gray_level_difference':maximum,
                      'changed_pixel_fraction':changed/pixels}
    particles=[]
    for folder in [baseline,other]:
        particles.append(pd.read_csv(folder/'measurement/particles_3d.csv').sort_values(['frame','raw_row_id']).reset_index(drop=True))
    pa,pb=particles
    measurement={'rows':[len(pa),len(pb)]}
    measurement['same_detection_keys']=pa[['frame','raw_row_id']].equals(pb[['frame','raw_row_id']])
    if measurement['same_detection_keys']:
        measurement['max_abs_differences']={key:float(np.max(np.abs(pa[key].to_numpy()-pb[key].to_numpy())))
                                            for key in ['seg_xc','seg_yc','slice','diameter_px','final_diameter_um']}
    tracks=[]
    for folder in [baseline,other]:tracks.append(pd.read_parquet(folder/'tracking/smoothed.parquet').sort_values(['track_id','frame']).reset_index(drop=True))
    ta,tb=tracks
    tracking={'rows':[len(ta),len(tb)],'same_track_frame_keys':ta[['track_id','frame']].equals(tb[['track_id','frame']])}
    if tracking['same_track_frame_keys']:
        tracking['max_abs_differences']={key:float(np.max(np.abs(ta[key].to_numpy()-tb[key].to_numpy()))) for key in ['x','y','z','diameter_px']}
    pred=[pd.read_parquet(folder/'inference/predictions.parquet') for folder in [baseline,other]]
    top=[pd.read_parquet(folder/'inference/top80.parquet') for folder in [baseline,other]]
    same_ids=set(pred[0].candidate_id)==set(pred[1].candidate_id)
    scores=None
    if same_ids:
        a=pred[0].set_index('candidate_id').sort_index();b=pred[1].set_index('candidate_id').sort_index()
        scores=float(np.max(np.abs(a.score-b.score)))
    predictions={'rows':[len(p) for p in pred],'same_candidates':same_ids,
                 'same_top80_order':top[0].candidate_id.tolist()==top[1].candidate_id.tolist(),
                 'same_top80_set':set(top[0].candidate_id)==set(top[1].candidate_id),
                 'max_score_difference':scores}
    criteria={
        'images_within_one_gray_level':all(v['maximum_gray_level_difference']<=1 for v in images.values()),
        'at_most_0p1_percent_pixels_changed':all(v['changed_pixel_fraction']<=.001 for v in images.values()),
        'same_detection_keys':measurement['same_detection_keys'],
        'same_focus_planes':measurement.get('max_abs_differences',{}).get('slice',float('inf'))==0,
        'particle_xy_diameter_within_0p05_px':all(measurement.get('max_abs_differences',{}).get(k,float('inf'))<=.05 for k in ['seg_xc','seg_yc','diameter_px']),
        'same_track_frame_keys':tracking['same_track_frame_keys'],
        'track_coordinates_within_0p05_grid_units':all(v<=.05 for v in tracking.get('max_abs_differences',{'missing':float('inf')}).values()),
        'same_final_candidate_set':same_ids,'same_ordered_top80':predictions['same_top80_order'],
        'candidate_scores_within_1e_minus4':scores is not None and scores<=1e-4,
        'same_model':all(s['execution']['model_json_sha256']==sha(MODEL/'model.json') for s in stages.values())}
    result={'status':'PASS' if all(criteria.values()) else 'DIFFERENT','label':label,'frames':128,
            'images':images,'measurement':measurement,'tracking':tracking,'predictions':predictions,
            'criteria':criteria,'scope':'Bounded device-equivalence check; no retraining and no scientific parameter changes.',
            'checked_at':stamp()}
    save(ROOT/f'distributed/probe-{label}-verification.json',result)
    print(json.dumps(result,indent=2));return all(criteria.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    raise SystemExit(0 if compare(a.label) else 2)
