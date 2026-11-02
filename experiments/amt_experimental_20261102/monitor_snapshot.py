"""Bounded, read-only telemetry; usable on either host without scientific imports."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import io
import json
import shutil
import subprocess

STAGES=['background','minip','measurement','tracking','inference','assets']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def alive(pid,expected=None):
    try:
        root=Path('/proc')/str(int(pid))
        if (root/'stat').read_text().split(') ')[1][0]=='Z':return False
        return not expected or expected in (root/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,ValueError,TypeError):return False


def tail(path,limit=65536):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-limit));raw=stream.read(limit)
        lines=raw.decode(errors='replace').splitlines()
        if size>limit:lines=lines[1:]
        if raw and not raw.endswith(b'\n'):lines=lines[:-1]
        return lines
    except OSError:return []


def modified(path):
    try:return dt.datetime.fromtimestamp(Path(path).stat().st_mtime,dt.timezone.utc).isoformat()
    except OSError:return None


def numbered_files(folder,suffix):
    try:paths=[p for p in Path(folder).iterdir() if p.suffix==suffix and p.stem.isdigit()]
    except OSError:return 0,None
    return len(paths),modified(max(paths,key=lambda p:int(p.stem))) if paths else None


def last_frame(path):
    for row in reversed(list(csv.reader(tail(path,16384)))):
        try:return int(row[0])+1
        except (ValueError,IndexError):continue
    return 0


def hardware():
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=8)
        rows=[]
        for r in csv.reader(io.StringIO(output)):
            index,name,used,total,util=[v.strip() for v in r]
            rows.append({'index':int(index),'name':name,'used_mib':int(used),'total_mib':int(total),'utilization':int(util)})
        return {'available':True,'gpus':rows,'observed_at':now()}
    except (OSError,ValueError,subprocess.SubprocessError):return {'available':False,'gpus':[],'observed_at':now()}


def scene_snapshot(root,n,stage=None,limit=10917,folder=None,worker_log=None):
    root=Path(root);folder=Path(folder) if folder else root/f'scenes/scene{n:04d}'
    markers={s:read(folder/f'stages/{s}.json') for s in STAGES}
    completed=[s for s,v in markers.items() if v and v.get('status')=='PASS']
    progress=None
    if stage=='background':
        counts=[numbered_files(folder/f'prepared/01_bkgrem/{c}H001S{n:04d}','.png') for c in ['C001','C002']]
        progress={'done':sum(x[0] for x in counts),'total':2*limit,'unit':'枚','label':'背景除去画像',
                  'output_at':max((x[1] for x in counts if x[1]),default=None)}
    elif stage=='minip':
        count,mtime=numbered_files(folder/'minip','.json')
        progress={'done':count,'total':limit,'unit':'フレーム','label':'照合済みMinIP','output_at':mtime}
    elif stage=='measurement':
        p=folder/'measurement/particles_3d.csv';count=last_frame(p)
        if count:
            progress={'done':count,'total':limit,'unit':'フレーム','label':'粒子計測の最終出力フレーム',
                      'note':'CSVの出力位置。工程全体の検証は終了時に行います。','output_at':modified(p)}
        else:
            p=folder/'measurement/frame_stats.csv'
            progress={'done':last_frame(p),'total':limit,'unit':'フレーム','label':'計測の前処理（検出）',
                      'note':'検出の後に奥行き・粒径を計測します。','output_at':modified(p)}
    elif stage=='assets':
        try:paths=list((folder/'review/media').glob('*.json'))
        except OSError:paths=[]
        progress={'done':len(paths),'total':80,'unit':'件','label':'レビューGIF',
                  'output_at':max((modified(p) for p in paths),default=None)}
    elif stage=='inference' and worker_log:
        for line in reversed(tail(worker_log)):
            try:value=json.loads(line)
            except ValueError:continue
            if not isinstance(value,dict) or not isinstance(value.get('total'),(int,float)):continue
            if value.get('stage') in ['background','minip','review_assets']:continue
            key=next((k for k in ['completed','done','candidates','frames'] if isinstance(value.get(k),(int,float))),None)
            if key:
                progress={'done':value[key],'total':value['total'],'unit':'件','label':'推論中の処理件数',
                          'note':'候補生成・特徴抽出・推論を順に実行します。','output_at':value.get('time')};break
    if progress:
        progress['done']=min(progress['done'],progress['total'])
    manifest=read(folder/'review/manifest.json',{})
    ready=bool('assets' in completed and manifest.get('status')=='PASS' and not manifest.get('smoke_only')
               and manifest.get('budget')==80 and len(manifest.get('candidates',[]))==80)
    return {'scene':n,'completed_stages':completed,'review_ready':ready,'progress':progress,
            'stage_seconds':{s:round(v['seconds']) for s,v in markers.items() if v and 'seconds' in v}}


def snapshot(root,scenes):
    root=Path(root);rows=[]
    for n in scenes:
        receipt=read(root/f'distributed/jobs/kit-scene{n:04d}-receipt.json',{})
        log=root/f'distributed/jobs/kit-scene{n:04d}-receipt.log'
        rows.append({**scene_snapshot(root,n,receipt.get('stage'),worker_log=log),
                     'stage':receipt.get('stage'),'status':receipt.get('status','pending'),
                     'process_alive':alive(receipt.get('pid'),'distributed_worker.py')})
    disk=shutil.disk_usage(root)
    return {'observed_at':now(),'hardware':hardware(),'scenes':rows,'disk_free_gib':round(disk.free/2**30,1)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scenes',type=int,nargs='*',default=[])
    a=p.parse_args();print(json.dumps(snapshot(a.root,a.scenes)))
