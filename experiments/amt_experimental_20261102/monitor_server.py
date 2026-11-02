"""Local monitoring page: telemetry is independent of the scientific workers."""
from pathlib import Path
import argparse
import concurrent.futures
import datetime as dt
import json
import shlex
import shutil
import sqlite3
import subprocess
import threading
import time
from flask import Flask,abort,jsonify,send_file
from monitor_snapshot import STAGES,alive,hardware,now,read,scene_snapshot

SCRIPT=Path(__file__).resolve().parent
ROOT=SCRIPT.parents[2]/'audit/revision-experimental-20261102'
SCENES=[12,13,15,17,18,20,21,22,24,25,26,28,29,30]
PREFIX='/experimental-monitor'


def age(value):
    try:return max(0,(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(value)).total_seconds())
    except (ValueError,TypeError):return None


def remote_snapshot(scenes):
    remote='/home/dai/amt-distributed-20261102'
    argv=[remote+'/gpu-env/bin/python',remote+'/workspace/HDCD-materials/experiments/amt_experimental_20261102/monitor_snapshot.py',
          '--root',remote+'/workspace/audit/revision-experimental-20261102','--scenes',*map(str,scenes)]
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ConnectionAttempts=1',
                                 'daiPCatKIT',shlex.join(argv)],text=True,timeout=18)
    return json.loads(raw)


def review_counts(root):
    path=Path(root)/'reviews.sqlite3'
    if not path.exists():return {}
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)
    try:
        rows=db.execute('SELECT scene,label,count(*) FROM reviews GROUP BY scene,label').fetchall()
        result={}
        for n,label,count in rows:result.setdefault(n,{})[label]=count
        return result
    finally:db.close()


class Collector:
    def __init__(self,root=ROOT,remote=remote_snapshot):
        self.root=Path(root);self.remote=remote;self.last_remote=None;self.latest=None

    def collect(self):
        root=self.root;dist=root/'distributed';state=read(root/'campaign-state.json',{})
        jobs=read(dist/'assignments.json',{});counts=review_counts(root)
        remote_jobs=[j['scene'] for w,j in jobs.items() if w=='daiPCatKIT-gpu0' and j.get('status')=='running']
        connected=True
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future=pool.submit(self.remote,remote_jobs);gpu=hardware()
            try:self.last_remote=future.result()
            except (OSError,ValueError,subprocess.SubprocessError):connected=False
        remote=self.last_remote or {'scenes':[],'hardware':{'gpus':[]},'observed_at':None}
        remote_scenes={r['scene']:r for r in remote['scenes']}
        rows={n:scene_snapshot(root,n) for n in SCENES}
        workers=[];alerts=[]
        controller_alive=alive(state.get('pid'),'dispatch.py')
        stale=age(state.get('updated_at'))
        if state.get('status')!='PASS' and (not controller_alive or stale is None or stale>90):
            alerts.append({'level':'error','text':'割り当て制御の更新が停止しています。表示は最後に確認できた状態です。'})
        for worker in state.get('workers',[]):
            item={k:worker.get(k) for k in ['worker','status','scene','stage']};name=item['worker'];n=item['scene'];job=jobs.get(name,{})
            is_remote=name=='daiPCatKIT-gpu0';index=1 if name=='local-gpu1' else 0
            device_source=remote['hardware'] if is_remote else gpu
            item['gpu']=next((g for g in device_source.get('gpus',[]) if g['index']==index),None)
            item['telemetry_at']=remote['observed_at'] if is_remote else gpu['observed_at']
            item['connected']=connected if is_remote else True
            if is_remote and not connected:
                item['status']='connection_lost'
                alerts.append({'level':'warning','text':'daiPCatKITへの接続を確認できません。前回値を表示し、自動で再接続します。'})
            if n and job.get('status')=='running':
                receipt=read(Path(job.get('receipt','')),{});stage=receipt.get('stage',item['stage'])
                if is_remote and n in remote_scenes:
                    detail=remote_scenes[n];stage=detail.get('stage') or stage
                    if connected and detail.get('status')=='running' and not detail.get('process_alive'):
                        item['status']='FAILED'
                else:
                    detail=scene_snapshot(root,n,stage,worker_log=dist/f'{name}-scene{n:04d}.log')
                item['stage']=stage;item['progress']=detail['progress']
                item['elapsed_seconds']=round(age(job.get('started_at')) or 0)
                rows[n].update(completed_stages=detail['completed_stages'],progress=detail['progress'],worker=name,stage=stage)
                # Remote completion is not review-ready until verified import has finished locally.
                if not alive(job.get('pid'),'remote_task.py' if is_remote else 'distributed_worker.py'):
                    item['status']='PASS' if receipt.get('status')=='PASS' else 'FAILED'
            if name=='local-gpu1' and not n:
                probe=read(dist/'probe-local-gpu1.json',{})
                if probe.get('status')=='running' and alive(probe.get('pid'),'distributed_worker.py'):
                    item.update(status='device_validation',stage=probe.get('stage'))
                    item['progress']=scene_snapshot(root,12,probe.get('stage'),limit=128,
                        folder=dist/'probes/local-gpu1/scene0012',worker_log=dist/'probe-local-gpu1-resumed.log')['progress']
                elif (dist/'probe-local-gpu1-resumed.json').exists() and probe.get('status')=='FAILED':
                    item['status']='validation_failed'
            if item['status'] in ['FAILED','validation_failed']:
                alerts.append({'level':'error','text':('GPU動作確認' if item['status']=='validation_failed' else '系列処理')+'でエラーを記録しました。途中出力は保存されています。'})
            workers.append(item)
        for n,row in rows.items():
            row['counts']=counts.get(n,{})
            row['reviewed']=sum(row['counts'].values());row['budget']=80
        equivalence=read(dist/'code-equivalence.json',{})
        local_disk=shutil.disk_usage(root)
        result={'schema':1,'updated_at':now(),'refresh_seconds':30,'controller_updated_at':state.get('updated_at'),
                'controller_alive':controller_alive,'campaign_status':state.get('status','unknown'),
                'workers':workers,'scenes':list(rows.values()),'alerts':alerts,
                'ready_scenes':sum(r['review_ready'] for r in rows.values()),
                'reviewable':80*sum(r['review_ready'] for r in rows.values()),
                'reviewed':sum(r['reviewed'] for r in rows.values()),'budget':1120,
                'local_disk_free_gib':round(local_disk.free/2**30,1),'remote_disk_free_gib':remote.get('disk_free_gib'),
                'code_equivalence':{'status':equivalence.get('status'),'checked_at':equivalence.get('checked_at'),
                    'production_files':equivalence.get('production_files_verified'),
                    'holo_linker_sha256':equivalence.get('actual_holo_linker_sha256')}}
        self.latest=result;return result


def create_app(collector):
    app=Flask(__name__,static_folder=None)
    @app.after_request
    def headers(response):
        response.headers.update({'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
            'Referrer-Policy':'no-referrer','X-Robots-Tag':'noindex, nofollow',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
        return response
    @app.get(PREFIX+'/')
    def index():return send_file(SCRIPT/'templates/monitor.html')
    @app.get(PREFIX+'/assets/<name>')
    def asset(name):
        if name not in ['monitor.css','monitor.js']:abort(404)
        return send_file(SCRIPT/'static'/name)
    @app.get(PREFIX+'/api/state')
    def state():
        if collector.latest is None:return jsonify({'status':'starting'}),503
        return jsonify(collector.latest)
    return app


def main(port):
    collector=Collector();output=ROOT/'web';output.mkdir(exist_ok=True)
    def watch():
        while True:
            try:
                value=collector.collect();temp=output/'monitor-state.partial';temp.write_text(json.dumps(value,ensure_ascii=False));temp.replace(output/'monitor-state.json')
            except Exception as exc:print(now(),'Telemetry collection failed:',type(exc).__name__,str(exc),flush=True)
            time.sleep(30)
    threading.Thread(target=watch,daemon=True).start()
    create_app(collector).run(host='127.0.0.1',port=port,threaded=True,use_reloader=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8771);a=p.parse_args();main(a.port)
