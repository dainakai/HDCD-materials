"""128-frame integration check; never used as the scientific performance result."""
from common import *
import os
import subprocess
import sys
import time
import yaml
import torch
from ra3d.tracks import read_tracks
from ra3d.features import generate_candidates,static_features,image_features
from ra3d.acquisition import Acquisition,Reconstructor
from ra3d.minip import MinIPReader
from minip_tracks import match_identity
from ra3d.strict_labels import attach_identity_labels
from ra3d.config import STATIC_FEATURES,IMAGE_FEATURES


def main():
    torch.set_num_threads(4)
    seed=260740;limit=128;out=ROOT/'integration-smoke';out.mkdir(exist_ok=True)
    began=time.monotonic()
    for script in ['minip_generate.py','measure.py']:
        subprocess.run([str(GPU_PYTHON),str(SCRIPT/script),'--seed',str(seed),'--limit',str(limit)],check=True)
    measured=ROOT/f'runs/measurement-smoke-{limit}/seed{seed}'
    source=ROOT/'upstream/production';linked=out/'tracking'
    if not (linked/'zsmooth_tuning_summary.json').exists():
        cmd=[str(GPU_PYTHON),str(source/'src/tracking/link_best_tiebreak_z010.py'),
             '--pipeline-run-dir',str(measured),'--out-dir',str(linked)]
        subprocess.run(cmd,cwd=source,check=True,env={**os.environ,'PATH':'/home/dai-server/.juliaup/bin:'+os.environ['PATH']})
    raw=read_tracks(linked/'trajectory3d')
    parts=[]
    for p in sorted((linked/'smoothedtrajs').glob('path_*.csv')):
        frame=pd.read_csv(p);frame['track_id']=int(p.stem.split('_')[-1]);parts.append(frame)
    smooth=pd.concat(parts,ignore_index=True).sort_values(['track_id','frame'],kind='stable').reset_index(drop=True)
    assert raw[['frame','track_id']].equals(smooth[['frame','track_id']])
    assert raw.frame.min()==0 and raw.frame.max()==127
    truth=read_table(truth_path(seed,'particles.parquet'))
    mapping=match_identity(raw,truth.loc[truth.frame<limit])
    wide=generate_candidates(raw);assert len(wide)>0
    event=events(seed);event=event.loc[event.frame.between(0,limit-1)].reset_index(drop=True)
    labeled,audit,_=attach_identity_labels(wide,event,mapping,z_tolerance_um=15000.)
    # A diverse deterministic sample tests both complete and incomplete pre-event observations.
    selection=np.unique(np.linspace(0,len(wide)-1,min(32,len(wide))).astype(int))
    sample=wide.iloc[selection].reset_index(drop=True)
    static=static_features(sample,smooth,linked/'smoothedtrajs')
    assert all(c in static for c in STATIC_FEATURES)
    config=yaml.safe_load((ROOT/f'config/seed{seed}-collision.yaml').read_text())
    frames=pd.read_csv(ROOT/f'config/seed{seed}-frames.csv').iloc[:limit]
    write_table(frames,out/'frames.csv');config['frames']='frames.csv'
    (out/'acquisition.yaml').write_text(yaml.safe_dump(config))
    a=Acquisition.load(out/'acquisition.yaml')
    features=image_features(static,smooth,Reconstructor(a,'cuda:0'))
    assert all(c in features for c in IMAGE_FEATURES)
    assert (features.rendered_pre_frames+features.skipped_pre_frames).eq(9).all()
    planar=smooth.copy();planar['z']=0.;planar['vz']=0.;planar['postOutProb']=0.
    for method in ['depth-free','minip-2d']:
        projection=image_features(static,planar,MinIPReader(minip_dir(seed),method))
        assert (projection.rendered_pre_frames+projection.skipped_pre_frames).eq(9).all()
    write_json({'status':'PASS','purpose':'Integration check only, excluded from performance tables and model fitting.',
                'frames':limit,'raw_rows':len(raw),'tracks':raw.track_id.nunique(),'mapped_rows':len(mapping),
                'candidates':len(wide),'features_checked':len(sample),'label_audit':audit,
                'seconds':time.monotonic()-began},out/'complete.json')
    emit(stage='integration_smoke',status='PASS')


if __name__=='__main__':main()
