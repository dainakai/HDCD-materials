"""Prepare private remote runtime and worker contracts without editing the engine."""
from common import *
import argparse
import csv
import io
import shutil
import subprocess
import tomllib

DIST=ROOT/'distributed'
HOST='daiPCatKIT'
REMOTE=Path('/home/dai/amt-distributed-20261102')


def rsync(source,target,extra=()):
    source=str(source).rstrip('/')+('/' if Path(source).is_dir() else '')
    subprocess.run(['rsync','-a','--mkpath','--exclude=__pycache__','--exclude=*.pyc',*extra,
                    source,f'{HOST}:{target}'],check=True)


def ssh(*args):
    import shlex
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,
                           shlex.join(list(map(str,args)))],check=True)


def local_config():
    DIST.mkdir(exist_ok=True)
    for index in [0,1]:
        save(DIST/f'gpu{index}.json',{'name':f'local-gpu{index}','cuda_visible_devices':str(index),
            'python':str(GPU_PYTHON),'raw':str(RAW),'julia':'/home/dai-server/.juliaup/bin/julia',
            'julia_depot_path':os.environ['JULIA_DEPOT_PATH'],'memory_fraction':None,
            'require_idle_gpu':index==1})
    save(DIST/'kit.json',{'name':'daiPCatKIT-gpu0','cuda_visible_devices':'0',
        'python':str(REMOTE/'gpu-env/bin/python'),'raw':str(REMOTE/'raw'),
        'julia':'/home/dai/.juliaup/bin/julia',
        'julia_depot_path':str(REMOTE/'julia-depot')+':/home/dai/.julia','memory_fraction':None})
    save(DIST/'remote-contract.json',{'host':HOST,'root':str(REMOTE),'owner_run_id':RUN_ID,
        'local_workspace':str(WORKSPACE),'remote_workspace':str(REMOTE/'workspace'),
        'source_identity':read(ROOT/'processing-source-identity.json'),
        'cleanup_scope':'Only this newly created task workspace and its verified input copies.'})


def prepare_remote():
    local_config()
    state=DIST/'remote-setup.json'
    save(state,{'status':'running','stage':'runtime_copy','started_at':stamp()})
    ssh('mkdir','-p',str(REMOTE/'runtime'),str(REMOTE/'workspace/HDCD-materials'),str(REMOTE/'julia-depot/packages'))
    rsync(DIST/'remote-contract.json',REMOTE/'owner-contract.json')
    rsync('/home/dai-server/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/',REMOTE/'runtime/python/')
    ssh(REMOTE/'runtime/python/bin/python3.12','-m','venv',REMOTE/'gpu-env')
    rsync(GPU_PYTHON.parent.parent/'lib/python3.12/site-packages/',REMOTE/'gpu-env/lib/python3.12/site-packages/')
    rsync(REPO/'.venv/lib/python3.12/site-packages/',REMOTE/'workspace/HDCD-materials/.venv/lib/python3.12/site-packages/')
    save(state,{'status':'running','stage':'engine_and_models','updated_at':stamp()})
    workspace=REMOTE/'workspace'
    for relative in ['experiments/amt_experimental_20261102','experiments/amt_full_20261101']:
        rsync(str(REPO/relative)+'/',workspace/'HDCD-materials'/relative/'',extra=['--exclude=results'])
    for relative in ['upstream/production','models/ours','src']:
        rsync(str(NUMERICAL/relative)+'/',workspace/'audit/revision-full-20261101'/relative/'')
    rsync(NUMERICAL/'finalized.json',workspace/'audit/revision-full-20261101/finalized.json')
    rsync(PREPARATION/'model-freeze.json',workspace/'audit/revision-review-20261101/experimental-preparation/model-freeze.json')
    rsync(ROOT/'processing-source-identity.json',workspace/'audit/revision-experimental-20261102/processing-source-identity.json')
    rsync(DIST/'kit.json',workspace/'audit/revision-experimental-20261102/distributed/kit.json')
    manifest=tomllib.loads((SOURCE/'Manifest.toml').read_text())['deps']
    roots=['CSV','DataFrames','JSON','ProgressMeter','SavitzkyGolay'];closure=set()
    def visit(name):
        if name in closure:return
        closure.add(name)
        for entry in manifest.get(name,[]):
            for dep in entry.get('deps',[]):visit(dep)
    for name in roots:visit(name)
    for name in sorted(closure):
        folder=Path('/home/dai-server/.julia/packages')/name
        if folder.exists():rsync(str(folder)+'/',REMOTE/'julia-depot/packages'/name/'')
    check='using Pkg; Pkg.activate(ARGS[1]); using CSV, DataFrames, JSON, ProgressMeter, SavitzkyGolay; println("AMT_JULIA_DEPENDENCIES_OK ",VERSION)'
    ssh('env','JULIA_PKG_OFFLINE=true','JULIA_DEPOT_PATH='+str(REMOTE/'julia-depot')+':/home/dai/.julia',
        '/home/dai/.juliaup/bin/julia','--startup-file=no','-e',check,workspace/'audit/revision-full-20261101/upstream/production')
    save(state,{'status':'PASS','completed_at':stamp(),'python_version':'3.12.12',
                'torch_version':'2.12.0+cu130','julia_version':'1.12.6','required_julia_packages':sorted(closure),
                'remote_root':str(REMOTE),'source_identity':read(ROOT/'processing-source-identity.json')})


def probe_root(label,n=12,limit=128):
    root=DIST/'probes'/label/f'scene{n:04d}'
    assert not (root/'stages').exists(), 'Do not overwrite a started probe'
    shutil.copytree(scene(n)/'input',root/'input',dirs_exist_ok=True)
    shutil.copytree(scene(n)/'prepared/02_calibration',root/'prepared/02_calibration',dirs_exist_ok=True)
    shutil.copy2(scene(n)/'prepared/variables.yaml',root/'prepared/variables.yaml')
    frames=root/'input/frames.csv'
    rows=list(csv.DictReader(frames.open()))[:limit]
    for row in rows:
        for key in ['primary','secondary']:row[key]=row[key].replace(str(scene(n)),str(root))
    with frames.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    acquisition=root/'input/acquisition.yaml'
    acquisition.write_text(acquisition.read_text().replace(str(scene(n)),str(root)))
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['local','remote','probe']);parser.add_argument('--label',default='local-gpu1')
    args=parser.parse_args()
    if args.mode=='local':local_config()
    elif args.mode=='remote':prepare_remote()
    else:print(probe_root(args.label))
