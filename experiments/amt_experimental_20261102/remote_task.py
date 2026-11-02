"""Move one immutable scene to KIT, run it, verify outputs, then publish locally."""
from common import *
from prepare_distributed import DIST,HOST,REMOTE,rsync
import argparse
import csv
import fcntl
import shutil
import shlex
import subprocess
import time
import traceback


def call_remote(mode,request):
    argv=[str(REMOTE/'gpu-env/bin/python'),str(REMOTE/'workspace/HDCD-materials/experiments/amt_experimental_20261102/remote_control.py'),
          mode,'--request',str(request)]
    text=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join(argv)],text=True)
    return json.loads(text.strip().splitlines()[-1])


def merge(source,target):
    """Move new output without overwriting any different pre-existing file."""
    if not target.exists():source.rename(target);return
    if source.is_dir():
        assert target.is_dir()
        for path in source.iterdir():merge(path,target/path.name)
    else:assert sha(source)==sha(target),(source,target)


def execute(n,probe=False,profile='all'):
    limit=128 if probe else 0
    assert profile in ['all','background'] or probe
    label='kit' if profile=='all' else 'kit-'+profile
    jobid=label+'-probe' if probe else f'{label}-scene{n:04d}'
    state=DIST/f'{jobid}.json'
    remote_base=REMOTE/'workspace/audit/revision-experimental-20261102'
    remote_root=remote_base/(f'distributed/probes/{label}/scene{n:04d}' if probe else f'scenes/scene{n:04d}')
    configdir=DIST/'transfer-config'/jobid
    configdir.mkdir(parents=True,exist_ok=True)
    request_path=remote_base/f'distributed/jobs/{jobid}-request.json'
    remote_receipt=remote_base/f'distributed/jobs/{jobid}-receipt.json'
    started=time.monotonic()
    def status(stage,**details):
        save(state,{'status':'running','scene':n,'stage':stage,'jobid':jobid,'pid':os.getpid(),
                    'probe':probe,'updated_at':stamp(),**details})
        emit(worker='daiPCatKIT',scene=n,stage=stage,**details)
    try:
        status('stage_inputs')
        shutil.copytree(scene(n)/'input',configdir/'input',dirs_exist_ok=True)
        frames=configdir/'input/frames.csv'
        rows=list(csv.DictReader(frames.open()))
        if limit:rows=rows[:limit]
        for row in rows:
            for key in ['primary','secondary']:row[key]=row[key].replace(str(scene(n)),str(remote_root))
        with frames.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        acquisition=configdir/'input/acquisition.yaml'
        acquisition.write_text(acquisition.read_text().replace(str(scene(n)),str(remote_root)))
        for name in ['common.py','process.py','distributed_worker.py','gpu_child.py','remote_control.py','runtime_attestation.py']:
            rsync(SCRIPT/name,REMOTE/'workspace/HDCD-materials/experiments/amt_experimental_20261102'/name)
        rsync(DIST/'expected-runtime.json',remote_base/'distributed/expected-runtime.json')
        rsync(configdir/'input',remote_root/'input')
        rsync(scene(n)/'prepared/02_calibration',remote_root/'prepared/02_calibration')
        rsync(scene(n)/'prepared/variables.yaml',remote_root/'prepared/variables.yaml')
        if profile.startswith('canonical-'):
            canonical=ROOT/f'smoke/128/scene{n:04d}'
            for name in ['prepared/01_bkgrem','minip']:
                rsync(canonical/name,remote_root/name)
            stage_names=['background','minip']
            if profile=='canonical-inference':
                for name in ['measurement','tracking']:rsync(canonical/name,remote_root/name)
                stage_names+=['measurement','tracking']
            for stage in stage_names:rsync(canonical/f'stages/{stage}.json',remote_root/f'stages/{stage}.json')
        for camera in ['C001','C002']:
            raw_rows=list(csv.DictReader((scene(n)/f'input/{camera}-raw-manifest.csv').open()))
            if limit:raw_rows=raw_rows[:limit]
            names=[r['name'] for r in raw_rows]
            assert all('/' not in name and name.endswith('.bmp') for name in names)
            filelist=configdir/f'{camera}-transfer-files.txt';filelist.write_text('\n'.join(names)+'\n')
            status('transfer_raw',camera=camera,frames=len(names))
            rsync(RAW/f'00_img/{camera}H001S{n:04d}',REMOTE/f'raw/00_img/{camera}H001S{n:04d}',extra=['--files-from='+str(filelist)])
        request={'run_id':RUN_ID,'scene':n,'limit':limit,'root':str(remote_root),'receipt':str(remote_receipt),
                 'stage':{'all':'all','background':'background','canonical-measurement':'measurement','canonical-inference':'inference'}[profile],
                 'canonical_contract_sha256':sha(scene(n)/'input/contract.json'),
                 'canonical_acquisition_sha256':sha(scene(n)/'input/acquisition.yaml'),
                 'remote_acquisition_sha256':sha(acquisition),'canonical_workspace':str(WORKSPACE),
                 'remote_workspace':str(REMOTE/'workspace')}
        local_request=configdir/'request.json';save(local_request,request);rsync(local_request,request_path)
        launch=call_remote('launch',request_path)
        failures=0
        while True:
            try:
                record=call_remote('status',request_path);failures=0
            except (OSError,subprocess.CalledProcessError):
                failures+=1
                if failures>=12:raise
                status('reconnecting',remote_pid=launch['pid']);time.sleep(15);continue
            status(record.get('stage','remote_running'),remote_pid=launch['pid'])
            if record['status']=='PASS':break
            if record['status']=='FAILED':raise RuntimeError(json.dumps(record))
            time.sleep(15)
        status('verify_remote_outputs')
        output=call_remote('manifest',request_path)
        incoming=DIST/'incoming'/jobid;incoming.mkdir(parents=True,exist_ok=True)
        status('transfer_results')
        subprocess.run(['rsync','-a','--exclude=.worker.lock','--exclude=.stage.lock',
                        f'{HOST}:{remote_root}/',str(incoming)+'/'],check=True)
        assert sha(incoming/'.output-manifest.json')==output['manifest_sha256']
        manifest=read(incoming/'.output-manifest.json')
        for relative,expected in manifest['files'].items():assert sha(incoming/relative)==expected,relative
        assert sha(incoming/'input/contract.json')==sha(scene(n)/'input/contract.json')
        for name,expected in read(ROOT/'processing-source-identity.json')['files'].items():assert sha(SCRIPT/name)==expected,name
        if probe:
            destination=DIST/'probes'/label/f'scene{n:04d}'
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert not destination.exists();incoming.rename(destination)
        else:
            destination=scene(n)
            lock=(destination/'.stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                assert not any((destination/'stages').glob('*.json')), 'Scene already has processing outputs'
                stage_records={p.stem:read(p) for p in (incoming/'stages').glob('*.json')}
                expected={'background'} if profile=='background' else {'background','minip','measurement','tracking','inference','assets'}
                assert set(stage_records)==expected
                review=None
                if profile=='all':
                    review=read(incoming/'review/manifest.json')
                    assert review['budget']==80 and len(review['candidates'])==80 and not review['smoke_only']
                    (incoming/'review/manifest.json').unlink()
                for path in incoming.iterdir():
                    if path.name in ['input','stages','.output-manifest.json']:continue
                    merge(path,destination/path.name)
                save(destination/f'remote-{profile}-execution.json',{'request':request,'output_manifest':manifest,'imported_at':stamp()})
                for name,record in stage_records.items():
                    if name!='assets':save(destination/f'stages/{name}.json',record)
                if review is not None:
                    save(destination/'review/manifest.json',review)
                    save(destination/'stages/assets.json',stage_records['assets'])
            finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        imported={'status':'PASS','remote_output_manifest_sha256':output['manifest_sha256'],
                  'verified_files':len(manifest['files']),'imported_at':stamp(),'destination':str(destination)}
        ack=configdir/'imported.json';save(ack,imported);rsync(ack,remote_receipt.with_suffix('.imported.json'))
        # Cleanup is a separate, verified operation; the original HDD remains read-only.
        cleanup=call_remote('cleanup',request_path)
        save(state,{'status':'PASS','scene':n,'jobid':jobid,'probe':probe,'destination':str(destination),
                    'import':imported,'cleanup':cleanup,'seconds':time.monotonic()-started,'finished_at':stamp()})
    except BaseException as exc:
        save(state,{'status':'FAILED','scene':n,'jobid':jobid,'probe':probe,'error':str(exc),
                    'traceback':traceback.format_exc(),'failed_at':stamp()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--probe',action='store_true')
    p.add_argument('--profile',choices=['all','background','canonical-measurement','canonical-inference'],default='all')
    a=p.parse_args();assert a.scene in SCENES;execute(a.scene,a.probe,a.profile)
