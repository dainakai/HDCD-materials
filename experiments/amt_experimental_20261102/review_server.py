"""Phone review of frozen top-80 cohorts; no training labels or raw-file serving."""
from common import ROOT, SCRIPT, SCENES, RUN_ID, MODEL, sha, read, save, stamp, scene
import argparse
import csv
import io
import secrets
import sqlite3
from pathlib import Path
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session

PREFIX='/experimental-review'
LABELS=('collision','non_collision','ambiguous')


def create_app(root=ROOT, access=None):
    root=Path(root)
    config=root/'review-access.json'
    if access is None:
        if not config.exists():save(config,{'key':secrets.token_urlsafe(24),'session_secret':secrets.token_hex(32)})
        access=read(config)
    app=Flask(__name__,template_folder=str(SCRIPT/'templates'),static_folder=None)
    app.config.update(SECRET_KEY=access['session_secret'],MAX_CONTENT_LENGTH=16384,
                      SESSION_COOKIE_NAME='amt_experimental_review',SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Strict',SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_PATH=PREFIX)
    dbpath=root/'reviews.sqlite3'
    def connect():
        db=sqlite3.connect(dbpath,timeout=30);db.row_factory=sqlite3.Row
        return db
    with connect() as db:
        db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS cohorts(scene INTEGER PRIMARY KEY,run_id TEXT NOT NULL,model_sha256 TEXT NOT NULL,prediction_sha256 TEXT NOT NULL,manifest_sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews(candidate_id TEXT PRIMARY KEY,scene INTEGER NOT NULL,label TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL,asset_sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT,candidate_id TEXT NOT NULL,scene INTEGER NOT NULL,label TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL,asset_sha256 TEXT NOT NULL);
        ''')
    cache={}
    def cohort(n):
        path=root/f'scenes/scene{n:04d}/review/manifest.json'
        if not path.exists():return None
        digest=sha(path)
        if n in cache:
            if cache[n][0]!=digest:raise RuntimeError('Frozen cohort changed; use a new review workspace')
            return cache[n][1]
        value=read(path)
        if value['smoke_only'] or value['run_id']!=RUN_ID or value['budget']!=80 or len(value['candidates'])!=80:
            raise RuntimeError('Invalid main review cohort')
        candidates=value['candidates']
        if sorted(c['review_rank'] for c in candidates)!=list(range(1,81)) or len({c['candidate_id'] for c in candidates})!=80:
            raise RuntimeError('Invalid candidate ranks or identities')
        for item in candidates:
            p=(root/item['asset']).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p)!=item['asset_sha256']:raise RuntimeError('Invalid review image')
        identity=(n,RUN_ID,value['model_sha256'],value['prediction_sha256'],digest)
        with connect() as db:
            old=db.execute('SELECT * FROM cohorts WHERE scene=?',(n,)).fetchone()
            if old and tuple(old)!=identity:raise RuntimeError('Cohort fingerprint differs from saved review store')
            db.execute('INSERT OR IGNORE INTO cohorts VALUES(?,?,?,?,?)',identity)
        cache[n]=(digest,value)
        return value
    def candidate(n,key):
        data=cohort(n)
        if data is None:abort(404)
        return next((c for c in data['candidates'] if c['candidate_id']==key),None)
    @app.before_request
    def authorize():
        if request.path==PREFIX+'/' and request.args.get('key'):
            if not secrets.compare_digest(request.args['key'],access['key']):abort(403)
            session['access']=True;session['csrf']=secrets.token_hex(24)
            return redirect(PREFIX+'/')
        if not session.get('access'):abort(403)
        if request.method=='POST':
            if not secrets.compare_digest(request.headers.get('X-CSRF-Token',''),session.get('csrf','!')):abort(403)
            if not request.is_json:abort(415)
    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['X-Robots-Tag']='noindex, nofollow'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        return response
    @app.errorhandler(403)
    def forbidden(error):
        if request.path==PREFIX+'/':
            return '<!doctype html><html lang="ja"><meta name="viewport" content="width=device-width,initial-scale=1"><title>レビュー専用リンク</title><h1>レビュー専用リンクから開いてください</h1><p>この画面は専用リンクで保護されています。このスレッドでお渡ししたレビューURLから開くと、その後はこの端末で続けられます。</p><p><a href="/experimental-monitor/">処理状況を見る</a></p></html>',403
        return error
    @app.get(PREFIX+'/')
    def index():return render_template('review.html',csrf=session['csrf'])
    @app.get(PREFIX+'/assets/<name>')
    def assets(name):
        if name not in {'review.css','review.js'}:abort(404)
        return send_file(SCRIPT/'static'/name)
    @app.get(PREFIX+'/api/status')
    def status():
        rows=[]
        with connect() as db:
            for n in SCENES:
                data=cohort(n)
                counts={x:0 for x in LABELS}
                for item in db.execute('SELECT label,count(*) AS total FROM reviews WHERE scene=? GROUP BY label',(n,)):
                    counts[item['label']]=item['total']
                completed=[stage for stage in ['background','minip','measurement','tracking','inference','assets']
                           if (root/f'scenes/scene{n:04d}/stages/{stage}.json').exists()]
                rows.append({'scene':n,'ready':bool(data),'budget':80,'counts':counts,
                             'reviewed':sum(counts.values()),'completed_stages':completed})
        current=read(root/'campaign-state.json') if (root/'campaign-state.json').exists() else {'status':'not_started'}
        return jsonify({'scenes':rows,'budget_total':1120,'run_id':RUN_ID,
                        'processing':{k:current[k] for k in ['status','scene','stage','started_at','updated_at','completed_scenes'] if k in current}})
    @app.get(PREFIX+'/api/scenes/<int:n>')
    def events(n):
        data=cohort(n)
        if data is None:abort(404)
        with connect() as db:reviews={r['candidate_id']:dict(r) for r in db.execute('SELECT * FROM reviews WHERE scene=?',(n,))}
        rows=[]
        for value in data['candidates']:
            item={k:v for k,v in value.items() if k not in ['asset','asset_sha256']}
            item['movie_url']=PREFIX+f'/api/scenes/{n}/events/{value["candidate_id"]}/movie'
            item['review']=reviews.get(value['candidate_id']);rows.append(item)
        return jsonify({'events':rows,'scene':n})
    @app.get(PREFIX+'/api/scenes/<int:n>/events/<key>/movie')
    def movie(n,key):
        item=candidate(n,key)
        if item is None:abort(404)
        return send_file(root/item['asset'],mimetype='image/gif')
    @app.post(PREFIX+'/api/scenes/<int:n>/events/<key>/review')
    def review(n,key):
        item=candidate(n,key)
        if item is None:abort(404)
        body=request.get_json()
        if not isinstance(body,dict):abort(400)
        label=body.get('label');note=body.get('note','')
        if label not in LABELS or not isinstance(note,str) or len(note)>4000:abort(400)
        values=(key,n,label,note,stamp(),item['asset_sha256'])
        with connect() as db:
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET label=excluded.label,note=excluded.note,updated_at=excluded.updated_at,asset_sha256=excluded.asset_sha256',values)
            db.execute('INSERT INTO history(candidate_id,scene,label,note,updated_at,asset_sha256) VALUES(?,?,?,?,?,?)',values)
        return jsonify({'saved':True,'candidate_id':key,'label':label})
    @app.post(PREFIX+'/api/scenes/<int:n>/events/<key>/undo')
    def undo(n,key):
        item=candidate(n,key)
        if item is None:abort(404)
        with connect() as db:
            old=db.execute('SELECT * FROM reviews WHERE candidate_id=? AND scene=?',(key,n)).fetchone()
            if old:
                db.execute('INSERT INTO history(candidate_id,scene,label,note,updated_at,asset_sha256) VALUES(?,?,?,?,?,?)',
                           (key,n,'unreviewed',old['note'],stamp(),item['asset_sha256']))
                db.execute('DELETE FROM reviews WHERE candidate_id=? AND scene=?',(key,n))
        return jsonify({'saved':True,'candidate_id':key,'label':None})
    @app.get(PREFIX+'/export.csv')
    def export():
        with connect() as db:
            rows=[dict(r) for r in db.execute('SELECT c.run_id,c.model_sha256,c.prediction_sha256,r.* FROM reviews r JOIN cohorts c USING(scene) ORDER BY scene,candidate_id')]
        stream=io.StringIO();columns=['run_id','model_sha256','prediction_sha256','candidate_id','scene','label','note','updated_at','asset_sha256']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        return send_file(io.BytesIO(stream.getvalue().encode('utf-8-sig')),as_attachment=True,download_name='amt-new-experimental-reviews.csv',mimetype='text/csv')
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8770);args=parser.parse_args()
    create_app().run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)
