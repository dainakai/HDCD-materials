"""Verify frozen production code, actual Holo-Linker imports, and runtime identity."""
from common import *
import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import shutil

ADAPTER_FILES=['distributed_worker.py','runtime_attestation.py','remote_control.py','gpu_child.py']


def runtime_identity():
    verify_engine()
    production=read(REPO/'experiments/amt_full_20261101/production-source.json')['files']
    for name,expected in production.items():assert sha(SOURCE/name)==expected,('production source mismatch',name)
    pinned=read(ROOT/'processing-source-identity.json')['files']
    for name,expected in pinned.items():assert sha(SCRIPT/name)==expected,('processing mismatch',name)
    sys.path.insert(0,str(SOURCE/'src/tracking/laptrack'))
    imports={}
    for name in ['holo_linker','two_stage_lap','depth3d_online_link']:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        expected=(SOURCE/'src/tracking/laptrack'/f'{name}.py').resolve()
        assert path==expected,('shadowed tracking module',name,path,expected)
        imports[name]={'path':str(path),'relative_path':str(path.relative_to(SOURCE)),'sha256':sha(path)}
    modules={}
    for name in ['torch','torchvision','torch_tensorrt','tensorrt','triton','numpy','pandas','scipy','sklearn','ultralytics','cv2','PIL','pyarrow','xgboost','yaml','joblib']:
        if importlib.util.find_spec(name) is None:
            modules[name]={'available':False};continue
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        modules[name]={'available':True,'version':str(getattr(module,'__version__','unknown')),'path':str(path),'entrypoint_sha256':sha(path)}
    import torch
    julia_script='using CSV, DataFrames, JSON, ProgressMeter, SavitzkyGolay; mods=[CSV,DataFrames,JSON,ProgressMeter,SavitzkyGolay]; JSON.print(Dict("version"=>string(VERSION),"packages"=>Dict(string(nameof(m))=>Dict("version"=>string(Base.pkgversion(m)),"path"=>pathof(m)) for m in mods)))'
    julia_executable=shutil.which('julia') or '/home/dai-server/.juliaup/bin/julia'
    julia=json.loads(subprocess.check_output([julia_executable,'--startup-file=no','--history-file=no',
                     '--project='+str(SOURCE),'-e',julia_script],text=True))
    for package in julia['packages'].values():package['entrypoint_sha256']=sha(package['path'])
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True).strip().splitlines()
    result={'status':'PASS','production_files_verified':len(production),
        'production_manifest_sha256':sha(REPO/'experiments/amt_full_20261101/production-source.json'),
        'engine_manifest_sha256':sha(REPO/'experiments/amt_full_20261101/engine-manifest.json'),
        'processing_files':pinned,'actual_tracking_imports':imports,'modules':modules,
        'execution_adapter_files':{name:sha(SCRIPT/name) for name in ADAPTER_FILES},
        'numerical_helper_files':{'minip_generate.py':sha(REPO/'experiments/amt_full_20261101/minip_generate.py')},
        'python':platform.python_version(),'python_build':list(platform.python_build()),'julia':julia,
        'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu_inventory':driver,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_json_sha256':sha(MODEL/'model.json'),'checked_at':stamp()}
    return result


def canonical(record):
    return {'production_manifest_sha256':record['production_manifest_sha256'],
            'engine_manifest_sha256':record['engine_manifest_sha256'],
            'processing_files':record['processing_files'],'model_json_sha256':record['model_json_sha256'],
            'execution_adapter_files':record['execution_adapter_files'],
            'numerical_helper_files':record['numerical_helper_files'],
            'tracking':{k:{x:v[x] for x in ['relative_path','sha256']} for k,v in record['actual_tracking_imports'].items()},
            'modules':{k:{x:v.get(x) for x in ['available','version','entrypoint_sha256']} for k,v in record['modules'].items()},
            'python':record['python'],'python_build':record['python_build'],'torch_cuda':record['torch_cuda'],'cudnn':record['cudnn'],
            'julia':{'version':record['julia']['version'],'packages':{
                k:{x:v[x] for x in ['version','entrypoint_sha256']} for k,v in record['julia']['packages'].items()}}}


def check(expected=None):
    identity=runtime_identity()
    expected=expected or ROOT/'distributed/expected-runtime.json'
    assert Path(expected).is_file(), 'Pinned runtime identity is missing'
    assert canonical(identity)==canonical(read(expected)), 'Loaded runtime differs from the pinned local GPU runtime'
    identity['matches_expected_runtime']=True
    return identity


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--create-expected',action='store_true');p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.config:
        config=read(a.config);os.environ['CUDA_VISIBLE_DEVICES']=config['cuda_visible_devices']
        os.environ['PATH']=str(Path(config['julia']).parent)+':'+os.environ['PATH']
        os.environ['JULIA_DEPOT_PATH']=config['julia_depot_path']
    record=runtime_identity() if a.create_expected else check();save(a.output,record)
    print(json.dumps({'status':record['status'],'production_files_verified':record['production_files_verified'],
                      'actual_tracking_imports':record['actual_tracking_imports'],'python':record['python'],
                      'torch':record['modules']['torch']['version'],'matches_expected_runtime':record.get('matches_expected_runtime')}))
