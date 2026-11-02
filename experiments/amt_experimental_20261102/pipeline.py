"""Durable ordered experimental processing and immutable per-scene top-80 exports."""
from common import *
import argparse
import fcntl
import subprocess
import time
import traceback

STAGES=['background','minip','measurement','tracking','inference','assets']


def main(scenes, smoke):
    verify_engine()
    state=ROOT/(f'smoke-{smoke}-state.json' if smoke else 'campaign-state.json')
    lock=state.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_identity={name:sha(SCRIPT/name) for name in ['common.py','inventory.py','process.py','pipeline.py']}
    save(ROOT/('smoke-source-identity.json' if smoke else 'processing-source-identity.json'),
         {'run_id':RUN_ID,'files':source_identity,'model_json_sha256':sha(MODEL/'model.json'),
          'engine_manifest_sha256':sha(SCRIPT.parent/'amt_full_20261101/engine-manifest.json'),
          'recorded_at':stamp()})
    started=time.monotonic()
    try:
        for n in scenes:
            for stage in STAGES:
                assert {name:sha(SCRIPT/name) for name in source_identity}==source_identity, 'Processing code changed during the campaign'
                record={'status':'running','pid':os.getpid(),'scene':n,'stage':stage,'smoke_frames':smoke,
                        'started_at':stamp(),'seconds':time.monotonic()-started,'scenes':scenes}
                save(state,record)
                log=ROOT/f'logs/{"smoke-"+str(smoke)+"-" if smoke else ""}scene{n:04d}-{stage}.log'
                log.parent.mkdir(exist_ok=True)
                args=[str(GPU_PYTHON),str(SCRIPT/'process.py'),stage,'--scene',str(n)]
                if smoke:args+=['--smoke',str(smoke)]
                emit(**record,log=str(log))
                with log.open('a') as stream:
                    subprocess.run(args,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True,
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
        save(state,{'status':'PASS','pid':os.getpid(),'scenes':scenes,'smoke_frames':smoke,
                    'seconds':time.monotonic()-started,'finished_at':stamp(),
                    'review_status':'not_started','budget_per_scene':80})
    except BaseException as error:
        save(state,{'status':'FAILED','pid':os.getpid(),'scene':n,'stage':stage,'smoke_frames':smoke,
                    'error':str(error),'traceback':traceback.format_exc(),'failed_at':stamp()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',type=int,nargs='+',default=SCENES)
    parser.add_argument('--smoke',type=int,default=0);args=parser.parse_args()
    assert set(args.scenes)<=set(SCENES)
    main(args.scenes,args.smoke)
