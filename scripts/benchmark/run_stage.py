"""Run one dependency-ready preparation stage, with bounded CPU concurrency."""
from common import *
import argparse
import concurrent.futures
import subprocess
import sys
import time

def run(seed,stage,method,device,threads):
    dependencies=[]
    if stage=='pool' and method=='minip2d':dependencies=[tracking(seed,method)/'complete.json']
    if stage=='static':dependencies=[path(s,method,'pool.json') for s in set(TRAIN+[seed])]
    if stage=='score':dependencies=[path(s,method,'static.json') for s in set(TRAIN+[seed])]
    if stage=='images':dependencies=[path(seed,method,'score.json')]
    if dependencies:
        deadline=time.monotonic()+7200
        if not all(p.exists() for p in dependencies):emit(stage='await_dependencies',operation=stage,method=method,seed=seed)
        while not all(p.exists() for p in dependencies):
            if time.monotonic()>deadline:raise TimeoutError(str(dependencies))
            time.sleep(5)
    log=ROOT/f'logs/{method}-{stage}-{seed}.log'
    if stage=='tracks':
        command=[sys.executable,str(SCRIPT/'minip_tracks.py'),'--seed',str(seed)]
    else:
        command=[sys.executable,str(SCRIPT/'prepare.py'),stage,'--seed',str(seed),
                 '--method',method,'--device',device,'--threads',str(threads)]
    emit(stage='start',method=method,operation=stage,seed=seed,log=str(log))
    with log.open('a') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    emit(stage='finish',method=method,operation=stage,seed=seed,returncode=result.returncode)
    if result.returncode:raise RuntimeError(f'Failed: {log}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--method',required=True)
    p.add_argument('--jobs',type=int,default=3);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu');p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    a=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as executor:
        futures=[executor.submit(run,s,a.stage,a.method,a.device,a.threads) for s in a.seeds]
        for future in concurrent.futures.as_completed(futures):future.result()
    emit(status='PASS',operation=a.stage,method=a.method,seeds=a.seeds)
