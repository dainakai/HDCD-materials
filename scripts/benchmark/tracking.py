"""Link and smooth 3D particle measurements, then associate truth identities."""
from common import *
import argparse
import os
import subprocess
import time
from ra3d.tracks import read_tracks
from minip_tracks import match_identity
from threadpoolctl import threadpool_limits


def run(seed):
    out=tracking(seed,'ours');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():emit(stage='tracking_already_complete',seed=seed);return
    source=measurement(seed);assert (source/'complete.json').exists()
    origin=REPO/'reference/measurement'
    linked=out/'linking';began=time.monotonic()
    if not (linked/'zsmooth_tuning_summary.json').exists():
        command=[str(GPU_PYTHON),str(origin/'src/tracking/link_best_tiebreak_z010.py'),
                 '--pipeline-run-dir',str(source),'--out-dir',str(linked)]
        if (linked/'depth3d_link_diagnostics.json').exists():
            command=[JULIA,str(origin/'src/tracking/laptrack/zsmooth_new.jl'),
                     f'--run-dir={linked}','--selection=auto','--force']
        emit(stage='original_tracking',seed=seed,command=command)
        subprocess.run(command,cwd=origin,check=True,env=os.environ.copy())
    if not (out/'smoothing').exists():(out/'smoothing').symlink_to(linked,target_is_directory=True)
    raw=read_tracks(linked/'trajectory3d')
    # Use the same historical CSV parse for kinematic features, image focus and Fig.7.
    parts=[]
    for file in sorted((linked/'smoothedtrajs').glob('path_*.csv')):
        part=pd.read_csv(file);part['track_id']=int(file.stem.split('_')[-1]);parts.append(part)
    smooth=pd.concat(parts,ignore_index=True).sort_values(['track_id','frame'],kind='stable').reset_index(drop=True)
    assert raw[['track_id','frame']].equals(smooth[['track_id','frame']])
    assert raw.frame.min()==0 and raw.frame.max()==5999
    write_table(raw,out/'raw.parquet');write_table(smooth,out/'smoothed.parquet')
    truth=read_table(truth_path(seed,'particles.parquet'))
    mapping=match_identity(raw,truth);write_table(mapping,out/'mapping.parquet')
    write_json({'status':'PASS','seed':seed,'rows':len(raw),'tracks':raw.track_id.nunique(),
                'frames':raw.frame.nunique(),'mapped_rows':len(mapping),
                'input_sha256':sha256(source/'particles_3d.csv'),
                'truth_sha256':sha256(truth_path(seed,'particles.parquet')),
                'outputs':{n:sha256(out/n) for n in ['raw.parquet','smoothed.parquet','mapping.parquet']},
                'seconds':time.monotonic()-began},out/'complete.json')
    emit(stage='tracking_complete',seed=seed,seconds=time.monotonic()-began)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.seed)
