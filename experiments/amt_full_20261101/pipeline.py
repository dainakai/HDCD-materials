"""Durable six-sequence campaign; fail explicitly rather than mix old artifacts."""
from common import *
import argparse
import concurrent.futures
import fcntl
import os
import subprocess
import time
import traceback

STATE=ROOT/'campaign-state.json'


def execute(name,args,gpu=False):
    command=[str(GPU_PYTHON if gpu else CPU_PYTHON),str(SCRIPT/args[0]),*map(str,args[1:])]
    log=ROOT/f'logs/{name}.log';started=time.monotonic()
    emit(stage='start',name=name,command=command,log=str(log))
    write_json({'status':'running','name':name,'command':command,'log':str(log),
                'started_at':dt.datetime.now().astimezone().isoformat()},ROOT/f'jobs/{name}.json')
    with log.open('a') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                              env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'})
    record={'status':'PASS' if result.returncode==0 else 'FAILED','name':name,'command':command,
            'returncode':result.returncode,'log':str(log),'seconds':time.monotonic()-started,
            'finished_at':dt.datetime.now().astimezone().isoformat()}
    write_json(record,ROOT/f'jobs/{name}.json');emit(**record)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {log}')


def main(seeds):
    lock=(ROOT/'.campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    write_json({'status':'running','pid':os.getpid(),'seeds':seeds,'started_at':dt.datetime.now().astimezone().isoformat()},STATE)
    try:
        execute('initialize',['initialize.py'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as cpu:
            futures=[]
            for seed in seeds:
                for future in futures:
                    if future.done():future.result()
                execute(f'render-{seed}',['render.py','--seed',seed,'--chunk',1000])
                execute(f'minip-{seed}',['minip_generate.py','--seed',seed],gpu=True)
                execute(f'measurement-{seed}',['measure.py','--seed',seed],gpu=True)
                futures.append(cpu.submit(execute,f'tracking-{seed}',['tracking.py','--seed',seed]))
                futures.append(cpu.submit(execute,f'minip-tracks-{seed}',['minip_tracks.py','--seed',seed]))
            for future in concurrent.futures.as_completed(futures):future.result()
        # Every method is rebuilt from its own new tracks, with all five training scenes.
        for stage in ['pool','static','score']:
            for method in ['ours','depth_free','minip2d']:
                execute(f'{method}-{stage}',['run_stage.py',stage,'--method',method,'--jobs',3,'--threads',4])
        execute('ours-images',['run_stage.py','images','--method','ours','--jobs',1,'--device','cuda:0'],gpu=True)
        for method in ['depth_free','minip2d']:
            execute(f'{method}-images',['run_stage.py','images','--method',method,'--jobs',3])
        for method in ['ours','depth_free','minip2d']:
            execute(f'train-{method}',['train.py','--method',method,'--device','cuda:0'],gpu=True)
            execute(f'evaluate-{method}',['evaluate.py','--method',method,*(['--table4'] if method=='ours' else [])])
        execute('figures',['figures.py'])
        execute('figure7',['figure7.py'])
        execute('figure9',['figure9.py'],gpu=True)
        write_json({'status':'PASS','pid':os.getpid(),'seeds':seeds,'seconds':time.monotonic()-started,
                    'finished_at':dt.datetime.now().astimezone().isoformat()},STATE)
    except BaseException as exc:
        write_json({'status':'FAILED','pid':os.getpid(),'error':str(exc),'traceback':traceback.format_exc(),
                    'seconds':time.monotonic()-started,'failed_at':dt.datetime.now().astimezone().isoformat()},STATE)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=parser.parse_args();main(args.seeds)
