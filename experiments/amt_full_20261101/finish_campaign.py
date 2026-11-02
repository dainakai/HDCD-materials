"""Resume the failed illustration and finalization without changing fitted results."""
from common import *
from pipeline import execute
import fcntl
import os
import shutil
import traceback


def main():
    lock=(ROOT/'.campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((ROOT/'campaign-state.json').read_text())
    assert state['status']=='FAILED' and state['error'].startswith('figure9 failed;')
    jobs={p.stem:json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*.json')}
    assert all(j['status']=='PASS' for name,j in jobs.items() if name!='figure9')
    protected=[]
    for method in ['ours','depth_free','minip2d']:
        protected.extend((ROOT/f'models/{method}').glob('*'))
        protected.extend((ROOT/f'results/{method}').glob('*'))
    protected=[p for p in protected if p.is_file()]
    before={str(p.relative_to(ROOT)):sha256(p) for p in protected}
    stamp=dt.datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    recovery=ROOT/'recovery'/stamp;recovery.mkdir(parents=True)
    for name in ['campaign-state.json','jobs/figure9.json','logs/figure9.log','finalization-job.json']:
        p=ROOT/name
        if p.exists():
            target=recovery/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json({'status':'running','pid':os.getpid(),'seeds':SEEDS,
                'resuming_only':['figure9','finalize'],'resumed_at':dt.datetime.now().astimezone().isoformat()},ROOT/'campaign-state.json')
    try:
        execute('figure9',['figure9.py'],gpu=True)
        after={str(p.relative_to(ROOT)):sha256(p) for p in protected}
        assert before==after,'Fitted models or numerical results changed during illustration recovery'
        write_json({'status':'PASS','unchanged_model_and_result_files':before,
                    'original_failure':state,'recovery_action':'Bootstrap optional dependencies before importing pandas in figure9.py.'},recovery/'verification.json')
        write_json({'status':'PASS','pid':os.getpid(),'seeds':SEEDS,
                    'original_numerical_run_seconds':state['seconds'],
                    'recovery_record':str(recovery/'verification.json'),
                    'finished_at':dt.datetime.now().astimezone().isoformat()},ROOT/'campaign-state.json')
        execute('finalize',['finalize.py'])
        write_json({'status':'PASS','returncode':0,'recovered_after':'figure9 import-order correction'},ROOT/'finalization-job.json')
    except BaseException as exc:
        write_json({'status':'FAILED','pid':os.getpid(),'seeds':SEEDS,'error':str(exc),
                    'traceback':traceback.format_exc(),'recovery_record':str(recovery),
                    'failed_at':dt.datetime.now().astimezone().isoformat()},ROOT/'campaign-state.json')
        raise


if __name__=='__main__':main()
