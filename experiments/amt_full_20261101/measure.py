"""Use the original production measurement backend on newly generated inputs."""
from common import *
import argparse
import os
import subprocess
import time

SOURCE=ROOT/'upstream/production'


def run(seed,limit=0):
    out=measurement(seed) if not limit else ROOT/f'runs/measurement-smoke-{limit}/seed{seed}'
    marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        assert record['frames']==(limit or 6000)
        assert sha256(out/'particles_3d.csv')==record['particles_sha256']
        emit(stage='measurement_already_complete',seed=seed,limit=limit);return
    if not limit:
        assert (minip_dir(seed)/'complete.json').is_file()
        assert len(list(minip_dir(seed).glob('*.png')))==6000
    weights={'--yolo-weights':'models/yolo/best.pt',
             '--depth-checkpoint':'models/depth/depth_compare_robust_v1.pt',
             '--depth-fallback-checkpoint':'models/depth/depth_compare_best.pt',
             '--slice-diam-weights':'models/slice_diam/slice_diammodel_best.pt'}
    cmd=[str(GPU_PYTHON),str(SOURCE/'src/pipeline/run_pipeline_fused.py'),
         '--data-dir',str(GENERATION/f'seed{seed}'),'--image-dir',str(minip_dir(seed)),
         '--run-dir',str(out),'--limit',str(limit),'--device','cuda:0',
         '--glassplate-to-volume-um','20000','--depth-padlen','1536','--keep-intermediate-csv']
    for flag,name in weights.items():cmd.extend([flag,str(SOURCE/name)])
    started=time.monotonic();emit(stage='measurement_start',seed=seed,limit=limit,command=cmd)
    subprocess.run(cmd,cwd=SOURCE,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
    frame=pd.read_csv(out/'particles_3d.csv',usecols=['frame','file','slice','z_um','depth_um'])
    canonical=frame.file.map(lambda v:int(Path(v).stem)-1).astype('int64')
    assert np.array_equal(frame.frame,canonical)
    assert frame.frame.nunique()==(limit or 6000)
    assert np.allclose(frame.z_um,frame['slice']*100.,rtol=0,atol=1e-8)
    assert np.allclose(frame.depth_um,74900.+(frame['slice']-1)*100.,rtol=0,atol=1e-8)
    metrics=json.loads((out/'fused_depth_slice_metrics.json').read_text())
    assert metrics['priter']==3,metrics
    write_json({'status':'PASS','seed':seed,'frames':int(frame.frame.nunique()),'rows':len(frame),
                'phase_iterations':3,'first_plane_um':74900,'last_plane_um':177200,'padding_px':1536,
                'minip_identity_sha256':sha256(minip_dir(seed)/'identity.json'),
                'weights':{n:sha256(SOURCE/n) for n in weights.values()},'command':cmd,
                'particles_sha256':sha256(out/'particles_3d.csv'),'seconds':time.monotonic()-started},marker)
    emit(stage='measurement_complete',seed=seed,limit=limit,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();run(a.seed,a.limit)
