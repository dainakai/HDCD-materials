"""Small remote launcher/status endpoint for the one explicitly authorized host."""
from common import *
import argparse
import csv
import signal
import shutil
import subprocess


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False


def main(mode,request):
    job=read(request);root=Path(job['root']);receipt=Path(job['receipt'])
    assert root.resolve().is_relative_to(ROOT.resolve()) and receipt.resolve().is_relative_to(ROOT.resolve())
    assert job['run_id']==RUN_ID
    config=read(ROOT/'distributed/kit.json')
    raw=Path(config['raw'])
    owner=read(raw.parent/'owner-contract.json');assert owner['owner_run_id']==RUN_ID
    launch=receipt.with_suffix('.launch.json')
    if mode=='launch':
        if launch.exists():
            state=read(receipt) if receipt.exists() else {}
            assert alive(read(launch)['pid']) or state.get('status')=='PASS', 'Previous remote attempt requires inspection'
            print(json.dumps(read(launch)));return
        contract=read(root/'input/contract.json')
        for camera in contract['cameras']:
            directory=raw/'00_img'/Path(camera['directory']).name
            rows=list(csv.DictReader((root/f'input/{camera["camera"]}-raw-manifest.csv').open()))
            if job['limit']:rows=rows[:job['limit']]
            assert sorted(p.name for p in directory.glob('*.bmp'))==sorted(r['name'] for r in rows)
            for row in rows:assert (directory/row['name']).stat().st_size==int(row['size_bytes'])
            for sample in camera['sample_content_hashes']:
                if sample['raw_index']<=len(rows):
                    assert sha(directory/rows[sample['raw_index']-1]['name'])==sample['sha256']
        verify_engine()
        argv=[config['python'],'-u',str(SCRIPT/'distributed_worker.py'),'--config',str(ROOT/'distributed/kit.json'),
              '--scene',str(job['scene']),'--root',str(root),'--limit',str(job['limit']),'--receipt',str(receipt),
              '--stage',job.get('stage','all')]
        log=receipt.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':child.pid,'argv':argv,'log':str(log),'started_at':stamp(),'request_sha256':sha(request)}
        save(launch,record);print(json.dumps(record))
    elif mode=='status':
        state=read(receipt) if receipt.exists() else {'status':'starting'}
        if launch.exists() and not alive(read(launch)['pid']) and state['status'] not in ['PASS','FAILED']:
            state={'status':'FAILED','error':'Remote process disappeared without a completion receipt'}
        print(json.dumps(state))
    elif mode=='manifest':
        assert read(receipt)['status']=='PASS'
        files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
               if p.is_file() and p.name not in ['.worker.lock','.stage.lock','.output-manifest.json']}
        result={'status':'PASS','scene':job['scene'],'limit':job['limit'],'files':files,
                'request_sha256':sha(request),'execution':read(receipt)}
        save(root/'.output-manifest.json',result)
        print(json.dumps({'status':'PASS','files':len(files),'manifest_sha256':sha(root/'.output-manifest.json')}))
    elif mode=='cleanup':
        ack=read(receipt.with_suffix('.imported.json'))
        assert ack['status']=='PASS' and ack['remote_output_manifest_sha256']==sha(root/'.output-manifest.json')
        assert not alive(read(launch)['pid'])
        for camera in ['C001','C002']:
            directory=raw/f'00_img/{camera}H001S{job["scene"]:04d}'
            assert directory.resolve().is_relative_to(raw.resolve());shutil.rmtree(directory)
        # Keep small receipts and input contracts; remove only imported bulky output.
        for name in ['prepared/01_bkgrem','minip','measurement','tracking','inference','review']:
            target=root/name
            assert target.resolve().is_relative_to(root.resolve())
            if target.exists():shutil.rmtree(target)
        print(json.dumps({'status':'PASS','cleaned_verified_copy':str(root)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','status','manifest','cleanup']);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args();main(a.mode,a.request)
