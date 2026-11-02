"""Durable scene queue for two local GPUs and the authorized daiPCatKIT worker."""
from common import *
import argparse
import fcntl
import signal
import subprocess
import time

DIST=ROOT/'distributed'


def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(') ')[1][0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False


def commandline(pid):
    try:return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


def gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    found={}
    for line in text.strip().splitlines():
        index,uuid,memory,util=map(str.strip,line.split(','));found[int(index)]={'uuid':uuid,'free_mib':int(memory),'utilization':int(util),'pids':[]}
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in text.strip().splitlines():
        if not line.strip():continue
        uuid,pid=map(str.strip,line.split(','))
        for gpu in found.values():
            if gpu['uuid']==uuid:gpu['pids'].append(int(pid))
    return found


def start(argv,log):
    with log.open('a') as stream:
        child=subprocess.Popen(list(map(str,argv)),cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return child.pid


def adopt(pid):
    path=DIST/'legacy-handoff.json'
    if path.exists():return read(path)
    assert str(SCRIPT/'pipeline.py') in commandline(pid), 'Refuse to suspend an unrelated or reused PID'
    snapshot=read(ROOT/'campaign-state.json')
    assert snapshot['pid']==pid and snapshot['status']=='running'
    os.kill(pid,signal.SIGSTOP)
    # SIGSTOP affects the serial controller only. Its current stage and all
    # descendants retain their GPU and finish writing their existing receipt.
    time.sleep(.1)
    snapshot=read(ROOT/'campaign-state.json')
    children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    children=[p for p in children if alive(p)]
    assert len(children)<=1,children
    record={'status':'draining','controller_pid':pid,'children':children,'scene':snapshot['scene'],
            'stage':snapshot['stage'],'prior_state':snapshot,'suspended_at':stamp(),
            'reason':'Transfer pending scenes to the dynamic dispatcher without interrupting the current scientific stage.'}
    save(path,record);return record


def drain_legacy(record):
    if record['status']=='complete':return
    if any(alive(pid) for pid in record['children']):return
    marker=scene(record['scene'])/f'stages/{record["stage"]}.json'
    assert marker.exists() and read(marker)['status']=='PASS', 'Legacy stage ended without a verified completion'
    pid=record['controller_pid']
    if alive(pid):
        assert str(SCRIPT/'pipeline.py') in commandline(pid)
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    record.update(status='complete',drained_at=stamp(),completed_stage_sha256=sha(marker))
    save(DIST/'legacy-handoff.json',record)


def main(legacy_pid=None,plan=False):
    verify_engine();DIST.mkdir(exist_ok=True)
    lock=(DIST/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if plan:
        print(json.dumps({'completed':[n for n in SCENES if (scene(n)/'review/manifest.json').exists()],
                          'local_gpus':gpus(),'remote_setup':read(DIST/'remote-setup.json')['status']}));return
    from runtime_attestation import canonical
    expected=read(DIST/'expected-runtime.json')
    for name in ['local-gpu1-runtime.json','kit-runtime.json']:
        record=read(DIST/name)
        assert record['status']=='PASS' and canonical(record)==canonical(expected), name
    legacy=adopt(legacy_pid) if legacy_pid else read(DIST/'legacy-handoff.json')
    path=DIST/'assignments.json'
    jobs=read(path) if path.exists() else {}
    idle_since=None
    runtime_sources={name:sha(SCRIPT/name) for name in ['distributed_worker.py','remote_control.py','remote_task.py','dispatch.py','runtime_attestation.py']}
    save(DIST/'dispatcher-process.json',{'pid':os.getpid(),'command':[str(CPU_PYTHON),str(Path(__file__))],
        'runtime_source_hashes':runtime_sources,'started_at':stamp(),'workers':['local-gpu0','local-gpu1','daiPCatKIT-gpu0']})
    while True:
        for name,expected_hash in runtime_sources.items():assert sha(SCRIPT/name)==expected_hash, name
        policy=read(DIST/'execution-policy.json')
        background_only=policy['additional_gpu_stages']==['background']
        drain_legacy(legacy)
        hardware=gpus();now=time.monotonic()
        if not hardware[1]['pids'] and hardware[1]['free_mib']>=10000:
            if idle_since is None:idle_since=now
        else:idle_since=None
        gpu1_idle=idle_since is not None and now-idle_since>=60
        # The failed memory-contention probe was archived. Resume its missing
        # stages only after a stable idle interval, without changing batch size.
        gpu1_admission=DIST/'probe-local-gpu1-verification.json'
        gpu1_probe=DIST/'probe-local-gpu1.json'
        if not background_only and not gpu1_admission.exists() and gpu1_idle:
            probe=read(gpu1_probe) if gpu1_probe.exists() else {}
            if probe.get('status')=='PASS':
                subprocess.run([str(CPU_PYTHON),str(SCRIPT/'check_device_probe.py'),'--label','local-gpu1'],check=False)
            elif not alive(probe.get('pid',0)) and not (DIST/'probe-local-gpu1-resumed.json').exists():
                args=[GPU_PYTHON,'-u',SCRIPT/'distributed_worker.py','--config',DIST/'gpu1.json','--scene','12',
                      '--root',DIST/'probes/local-gpu1/scene0012','--limit','128','--receipt',gpu1_probe]
                pid=start(args,DIST/'probe-local-gpu1-resumed.log')
                save(DIST/'probe-local-gpu1-resumed.json',{'pid':pid,'started_at':stamp()});idle_since=None
        kit_probe=DIST/'kit-probe.json';kit_admission=DIST/'probe-kit-verification.json'
        if kit_probe.exists() and read(kit_probe)['status']=='PASS' and not kit_admission.exists():
            subprocess.run([str(CPU_PYTHON),str(SCRIPT/'check_device_probe.py'),'--label','kit'],check=False)
        for worker,job in list(jobs.items()):
            if job['status']!='running':continue
            if alive(job['pid']):continue
            result=read(Path(job['receipt'])) if Path(job['receipt']).exists() else {}
            complete=(scene(job['scene'])/('stages/background.json' if job.get('profile')=='background' else 'review/manifest.json')).exists()
            job.update(status='PASS' if complete and result.get('status')=='PASS' else 'FAILED',finished_at=stamp())
            save(path,jobs)
        completed=[n for n in SCENES if (scene(n)/'review/manifest.json').exists()]
        reserved={job['scene'] for job in jobs.values() if job['status'] in ['running','FAILED']}
        if legacy['status']!='complete':reserved.add(legacy['scene'])
        acknowledged=policy.get('user_acknowledged_numerical_variance',False)
        kit_setup_ok=(DIST/'kit-runtime.json').exists() and read(DIST/'kit-runtime.json')['status']=='PASS'
        isolation=read(DIST/'kit-isolation-process.json') if (DIST/'kit-isolation-process.json').exists() else {}
        kit_busy=alive(isolation.get('pid',0))
        ready={'local-gpu0':legacy['status']=='complete',
               'local-gpu1':gpu1_idle and (background_only or (gpu1_admission.exists() and (
                   read(gpu1_admission)['status']=='PASS' or (acknowledged and read(gpu1_probe)['status']=='PASS')))),
               'daiPCatKIT-gpu0':kit_setup_ok and not kit_busy and (background_only or (kit_admission.exists() and (
                   read(kit_admission)['status']=='PASS' or (acknowledged and read(kit_probe)['status']=='PASS'))))}
        for worker in ['local-gpu0','daiPCatKIT-gpu0','local-gpu1']:
            if not ready[worker] or (worker in jobs and jobs[worker]['status'] in ['running','FAILED']):continue
            profile='background' if background_only and worker!='local-gpu0' else 'all'
            remaining=[n for n in SCENES if n not in completed and n not in reserved
                       and (profile!='background' or not (scene(n)/'stages/background.json').exists())]
            if not remaining:continue
            n=remaining[0]
            if worker=='daiPCatKIT-gpu0':
                receipt=DIST/(f'kit-background-scene{n:04d}.json' if profile=='background' else f'kit-scene{n:04d}.json')
                argv=[CPU_PYTHON,'-u',SCRIPT/'remote_task.py','--scene',str(n),'--profile',profile]
            else:
                index=0 if worker=='local-gpu0' else 1
                receipt=DIST/f'gpu{index}-scene{n:04d}.json'
                argv=[GPU_PYTHON,'-u',SCRIPT/'distributed_worker.py','--config',DIST/f'gpu{index}.json',
                      '--scene',str(n),'--receipt',receipt,'--stage',profile]
            job={'status':'starting','scene':n,'profile':profile,'receipt':str(receipt),'argv':list(map(str,argv)),'started_at':stamp()}
            jobs[worker]=job;save(path,jobs)
            job.update(pid=start(argv,DIST/f'{worker}-scene{n:04d}.log'),status='running')
            save(path,jobs);reserved.add(n)
            emit(stage='assigned',worker=worker,scene=n,pid=job['pid'])
        workers=[]
        for worker in ['local-gpu0','local-gpu1','daiPCatKIT-gpu0']:
            job=jobs.get(worker,{})
            item={'worker':worker,'status':job.get('status','waiting'),'scene':job.get('scene')}
            if job.get('receipt') and Path(job['receipt']).exists():item['stage']=read(Path(job['receipt'])).get('stage')
            if worker=='local-gpu0' and legacy['status']!='complete':
                item.update(status='running_existing_stage',scene=legacy['scene'],stage=legacy['stage'])
            if worker=='local-gpu1' and not job:
                resumed=read(DIST/'probe-local-gpu1-resumed.json') if (DIST/'probe-local-gpu1-resumed.json').exists() else {}
                item.update(status='device_validation' if alive(resumed.get('pid',0)) else
                            ('waiting_for_other_gpu_job' if hardware[1]['pids'] else 'device_validation'),
                            free_mib=hardware[1]['free_mib'])
            if worker=='daiPCatKIT-gpu0' and not job:
                item.update(status='device_validation',stage=read(kit_probe).get('stage') if kit_probe.exists() else 'setup')
            workers.append(item)
        state={'status':'PASS' if len(completed)==len(SCENES) else 'running','pid':os.getpid(),
               'mode':'dynamic_scene_queue','scenes':SCENES,'workers':workers,'completed_scenes':completed,
               'execution_policy':policy,
               'failed_workers':[w for w,j in jobs.items() if j['status']=='FAILED'],
               'budget_per_scene':80,'updated_at':stamp(),'legacy_stage_continues':legacy['status']!='complete'}
        active=next((w for w in workers if w.get('scene') not in completed and w.get('scene')),None)
        if active:state.update(scene=active['scene'],stage=active.get('stage'))
        save(ROOT/'campaign-state.json',state);save(DIST/'dispatcher-state.json',state)
        if state['status']=='PASS':break
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--legacy-pid',type=int);p.add_argument('--plan',action='store_true')
    a=p.parse_args();main(a.legacy_pid,a.plan)
