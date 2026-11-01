"""Publish aggregate progress and finished figures only to the report allowlist."""
from common import *
import argparse
import shutil
import time

REPORT=ROOT.parent/'revision-review-20261101'
PUBLIC=REPORT/'public'


def load_if(path):
    return json.loads(path.read_text()) if path.exists() else None


def update():
    state=load_if(ROOT/'campaign-state.json') or {'status':'not_started'}
    rows=[]
    for seed in SEEDS:
        data=GENERATION/f'seed{seed}'
        near={p.stem for p in (data/'01_bkgrem/C002H001S0011').glob('*.png')}
        far={p.stem for p in (data/'01_bkgrem/C001H001S0011').glob('*.png')}
        verified=load_if(ROOT/f'runs/render/seed{seed}/frames.json') or {}
        minip=minip_dir(seed)
        projections={p.stem for p in minip.glob('*.png')}
        markers={p.stem for p in minip.glob('*.json') if p.stem.isdigit()}
        measured=load_if(measurement(seed)/'complete.json')
        rows.append({'seed':seed,'split':'テスト' if seed==260745 else '学習',
                     'paired_holograms_saved':len(near&far),'paired_holograms_verified':len(verified.get('frames',{})),
                     'minip_verified':len(projections&markers),'measurement_frames':measured['frames'] if measured else 0,
                     'tracking_complete':(tracking(seed,'ours')/'complete.json').is_file(),
                     'minip_tracking_complete':(tracking(seed,'minip2d')/'complete.json').is_file()})
    jobs=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*.json')]
    active=[p['name'] for p in jobs if p['status']=='running']
    failed=[p['name'] for p in jobs if p['status']=='FAILED']
    methods=[]
    for method in ['ours','depth_free','minip2d']:
        summary=load_if(ROOT/f'results/{method}/summary.json')
        verified=load_if(ROOT/f'results/{method}/verification.json')
        methods.append({'method':method,'evaluated':bool(summary and verified and verified['status']=='PASS'),
                        'metrics':summary if verified and summary else None})
    figures=[];public_files={}
    for name,label in [('figure07','Fig.7：同じテスト系列の粒子計測'),('figure08','Fig.8：3手法の衝突検出性能'),
                       ('figure09','Fig.9：新しい候補の画像特徴'),('comparison','旧版との性能比較')]:
        source=ROOT/f'figures/{name}.png'
        if source.exists():
            relative=f'assets/full-{name}.png';target=PUBLIC/relative
            if not target.exists() or sha256(source)!=sha256(target):
                temporary=target.with_suffix('.partial');shutil.copyfile(source,temporary);temporary.replace(target)
            figures.append({'label':label,'path':relative});public_files[relative]=sha256(target)
    note={'status':state['status'],'updated_at':dt.datetime.now().astimezone().isoformat(),
          'current_jobs':active,'failed_jobs':failed,'sequences':rows,'total_frames':36000,
          'paired_holograms_verified':sum(r['paired_holograms_verified'] for r in rows),
          'paired_holograms_saved':sum(r['paired_holograms_saved'] for r in rows),
          'minip_verified':sum(r['minip_verified'] for r in rows),
          'measurement_frames':sum(r['measurement_frames'] for r in rows),
          'methods':methods,'figures':figures,'protocol':{'generation_depth_mm':102.3,'front_mm':74.9,
          'last_plane_mm':177.2,'phase_iterations':3,'minip_planes':513,'minip_padding':1536,'image_padding':2048},
          'interpretation':'検証済みの新生成データだけを集計。最終APは各手法の再学習と共通マッチング検証が完了してから表示。'}
    target=PUBLIC/'evidence/full-progress.json';write_json(note,target)
    public_files['evidence/full-progress.json']=sha256(target)
    manifest=load_if(REPORT/'public-manifest.json')
    manifest['files'].update(public_files)
    manifest['total_bytes']=sum((PUBLIC/name).stat().st_size for name in manifest['files'])
    write_json(manifest,REPORT/'public-manifest.json')
    write_json(note,ROOT/'progress.json')
    emit(stage='public_progress',status=note['status'],paired=note['paired_holograms_verified'],minip=note['minip_verified'],measurement=note['measurement_frames'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        update()
        if not a.watch:break
        time.sleep(30)
