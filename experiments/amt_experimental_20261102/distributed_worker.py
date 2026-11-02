"""Execution adapter around unchanged, hash-pinned scientific stage functions."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--scene',type=int,required=True)
    parser.add_argument('--stage',default='all')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=config['cuda_visible_devices']
    import common
    import process as engine
    engine.verify_engine()
    pinned=common.read(common.ROOT/'processing-source-identity.json')
    for name,expected in pinned['files'].items():
        assert common.sha(HERE/name)==expected,name
    engine.GPU_PYTHON=Path(config['python'])
    engine.RAW=Path(config['raw'])
    os.environ['PATH']=str(Path(config['julia']).parent)+':'+os.environ['PATH']
    os.environ['JULIA_DEPOT_PATH']=config['julia_depot_path']
    from runtime_attestation import check as check_runtime
    runtime=check_runtime()
    if config.get('memory_fraction'):
        engine.torch.cuda.set_per_process_memory_fraction(config['memory_fraction'],0)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':config['cuda_visible_devices']}

    def command(argv,log):
        argv=list(map(str,argv))
        if argv[0]=='/home/dai-server/.juliaup/bin/julia':argv[0]=config['julia']
        if argv[0]==config['python'] and config.get('memory_fraction'):
            argv=[argv[0],str(HERE/'gpu_child.py'),str(config['memory_fraction']),*argv[1:]]
        common.emit(stage='command',command=argv,log=str(log),worker=config['name'])
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as output:
            subprocess.run(argv,cwd=common.SOURCE,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    engine.command=command
    root=(args.root or common.scene(args.scene)).resolve()
    assert root.is_relative_to(common.ROOT.resolve())
    root.mkdir(parents=True,exist_ok=True)
    stages=engine.STAGES if args.stage=='all' else [args.stage]
    assert all(stage in engine.STAGES for stage in stages)
    # The scene-level lock prevents overlapping workers; stage locks also protect
    # the original serial process during an orderly handoff.
    lease=(root/'.worker.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    identity={'worker':config['name'],'host':socket.gethostname(),'pid':os.getpid(),
              'cuda_visible_devices':config['cuda_visible_devices'],'runtime_python':sys.executable,
              'adapter_sha256':common.sha(Path(__file__)),'frozen_process_sha256':common.sha(HERE/'process.py'),
              'model_json_sha256':common.sha(common.MODEL/'model.json'),
              'config_sha256':common.sha(args.config),'scene':args.scene,'limit':args.limit,
              'source_contract_sha256':common.sha(root/'input/contract.json')}
    identity['runtime']=runtime
    try:
        for stage in stages:
            for name,expected in pinned['files'].items():assert common.sha(HERE/name)==expected,name
            for name,expected in runtime['execution_adapter_files'].items():assert common.sha(HERE/name)==expected,name
            for name,expected in runtime['numerical_helper_files'].items():
                assert common.sha(common.REPO/'experiments/amt_full_20261101'/name)==expected,name
            engine.verify_engine()
            # Verify the complete frozen upstream code/weights manifest at every
            # stage boundary, in addition to the scientific engine's own check.
            production=common.read(common.REPO/'experiments/amt_full_20261101/production-source.json')['files']
            for name,expected in production.items():assert common.sha(common.SOURCE/name)==expected,name
            common.save(args.receipt,{**identity,'status':'running','stage':stage,'updated_at':common.stamp()})
            lock=(root/'.stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                marker=root/f'stages/{stage}.json'
                if marker.exists():
                    old=common.read(marker)
                    assert old['status']=='PASS' and old['input_contract_sha256']==identity['source_contract_sha256']
                    common.emit(stage=stage,scene=args.scene,status='ALREADY_COMPLETE')
                    continue
                previous=engine.STAGES.index(stage)
                if previous:assert common.read(root/f'stages/{engine.STAGES[previous-1]}.json')['status']=='PASS'
                engine.torch.set_num_threads(4);begin=time.monotonic()
                with engine.threadpool_limits(limits=4):
                    result=getattr(engine,stage)(args.scene,root,args.limit)
                common.save(marker,{'status':'PASS','scene':args.scene,'stage':stage,'smoke_frames':args.limit,
                    'finished_at':common.stamp(),'seconds':time.monotonic()-begin,'result':result,
                    'process_source_sha256':identity['frozen_process_sha256'],
                    'input_contract_sha256':identity['source_contract_sha256'],'execution':identity})
                common.emit(stage=stage,scene=args.scene,status='PASS',worker=config['name'])
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        common.save(args.receipt,{**identity,'status':'PASS','stages':stages,'seconds':time.monotonic()-started,'finished_at':common.stamp()})
    except BaseException as exc:
        common.save(args.receipt,{**identity,'status':'FAILED','stage':stage,'error':str(exc),
                                 'traceback':traceback.format_exc(),'failed_at':common.stamp()})
        raise


if __name__=='__main__':run()
