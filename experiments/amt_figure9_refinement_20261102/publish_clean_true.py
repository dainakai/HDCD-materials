#!/usr/bin/env python3
"""Document the rejected example's localization failure and publish alternatives."""
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from clean_true import OUT, REPO, ROOT, write
from frame_gallery import PREVIEW, PUBLIC, FINAL, sha, time_label
import numpy as np
import pandas as pd


def diagnose_old():
    raw=pd.read_parquet(ROOT/'runs/tracking/seed260745/raw.parquet')
    smooth=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    gt=pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',
        filters=[('frame','in',list(range(5519,5525))),('particle_id','in',[7607,7608,7683])])
    measurement=ROOT/'runs/measurement/seed260745/particles_3d.csv'
    with measurement.open() as f:header=f.readline()
    lines=subprocess.check_output(['rg','^(5519|5520|5521|5522|5523|5524),',str(measurement)],text=True)
    measured=pd.read_csv(io.StringIO(header+lines))
    points=[]
    for _,p in smooth.loc[smooth.track_id.isin([18638,18667])&smooth.frame.between(5519,5524)].iterrows():
        particle=7683 if p.frame>=5523 else (7607 if p.track_id==18638 else 7608)
        q=gt.loc[gt.frame.eq(p.frame)&gt.particle_id.eq(particle)].iloc[0]
        r=raw.loc[raw.frame.eq(p.frame)&raw.track_id.eq(p.track_id)].iloc[0]
        m=measured.loc[measured.frame.eq(p.frame)].copy()
        distance=np.hypot(m.seg_xc-r.x,m.seg_yc-r.y)
        hit=m.loc[distance.idxmin()]
        assert distance.min()<1e-5
        points.append(dict(frame=int(p.frame),track_id=int(p.track_id),truth_particle_id=particle,
            truth_xy_px=[q.x_px,q.y_px],raw_xy_px=[r.x,r.y],smooth_xy_px=[p.x,p.y],
            raw_xy_error_px=float(np.hypot(r.x-q.x_px,r.y-q.y_px)),
            smooth_xy_error_px=float(np.hypot(p.x-q.x_px,p.y-q.y_px)),
            smoothing_shift_px=float(np.hypot(p.x-r.x,p.y-r.y)),
            measurement={key:hit[key] for key in ['raw_row_id','raw_det_id','conf','xc','yc','seg_xc','seg_yc','w','h','diameter_method']}))
    red=[p for p in points if p['track_id']==18638]
    assert len(red)==4
    for p in red:
        assert p['measurement']['diameter_method']=='contrast_area_f70_bg90_full_gate2px_bbox_fallback'
        assert np.allclose(p['raw_xy_px'],[p['measurement']['xc'],p['measurement']['yc']],atol=1e-5)
    result=dict(status='PASS',points=points,
        all_points_mean_error_px=float(np.mean([p['smooth_xy_error_px'] for p in points])),
        all_points_max_error_px=max(p['smooth_xy_error_px'] for p in points),
        red_raw_error_range_px=[min(p['raw_xy_error_px'] for p in red),max(p['raw_xy_error_px'] for p in red)],
        red_smooth_error_range_px=[min(p['smooth_xy_error_px'] for p in red),max(p['smooth_xy_error_px'] for p in red)],
        cause='The raw detection box was small and near the particle edge. In all four red observations, the contrast-area centroid differed from the box center by >2px, triggering the documented bbox-center fallback. Raw centers already had 12–13px truth errors; smoothing did not correct them. The overlay accurately shows those saved measurements.',
        code='upstream/production/src/detection/raw_detections_depth_input.py:279-297',
        old_selection_failure='Particle diameter and projected-disk inclusion were insufficient illustration criteria. Center error must also be checked; this example is unsuitable as a clean tracking illustration.')
    write('old-a-diagnosis.json',result)
    return result


def main():
    protected={str(p):sha(p) for p in [FINAL/'fig09.pdf',FINAL/'selection.json',REPO.parent/'audit/revision-paper-replacements-20261102/paper-replacements.zip']}
    old=diagnose_old()
    cases=json.loads((OUT/'rendered-cases.json').read_text())
    rows=[f'<tr><td>前の(a)</td><td>286＋285→360</td><td>{old["all_points_mean_error_px"]:.2f}</td><td>{old["all_points_max_error_px"]:.2f}</td><td>中心位置の良い例として不採用</td></tr>']
    sections=[];summary=[]
    for c in cases:
        r=c['source'];e=r['truth_event'];cid=c['case_id'];v=int(r['vanished_track_id']);s=int(r['survivor_track_id'])
        red=[p['xy_error_px'] for p in r['observations'] if p['track_id']==v]
        title=f"{e['parent1_diameter_um']:.0f}＋{e['parent2_diameter_um']:.0f}→{e['child_diameter_um']:.0f} µm"
        rows.append(f'<tr><td><a href="#{cid}">{cid}</a></td><td>{title}</td><td>{r["mean_xy_error_px"]:.2f}</td><td>{r["max_xy_error_px"]:.2f}</td><td>赤点 {min(red):.2f}〜{max(red):.2f}</td></tr>')
        cards=[]
        for p in c['panels']:
            src=f"fig09-clean-a/{cid}/{p['filename']}"
            cards.append(f'<figure><figcaption><strong>{p["label"]}</strong> {time_label(p["delta"])}</figcaption><a href="{src}"><img src="{src}" width="256" height="256" alt="{p["label"]}：赤丸と青四角の実測軌跡を重ねた画像"></a><p>frame {p["frame"]}</p></figure>')
        summary.append(dict(case_id=cid,candidate_frame=int(r['candidate_frame']),display_reference_frame=int(r['display_reference_frame']),
            event_id=int(e['collision_event_id']),vanished_track_id=v,survivor_track_id=s,
            parent_diameters_um=[e['parent1_diameter_um'],e['parent2_diameter_um']],child_diameter_um=e['child_diameter_um'],
            trajectory_score=r['static_score'],image_feature_score=r['base_score'],
            mean_xy_error_px=r['mean_xy_error_px'],max_xy_error_px=r['max_xy_error_px'],
            red_xy_error_range_px=[min(red),max(red)],observations=r['observations']))
        sections.append(f'''<section id="{cid}"><h2>{cid} · {title}</h2>
<p>候補スコア {r['static_score']:.6f} → {r['base_score']:.6f}。赤点の位置誤差は {min(red):.2f}〜{max(red):.2f}画素。全10観測点では平均 {r['mean_xy_error_px']:.2f}、最大 {r['max_xy_error_px']:.2f}画素です。</p>
<div class="frames">{''.join(cards)}</div><p><a href="fig09-clean-a/{cid}-frames.png">6枚を横一列で開く</a></p></section>''')
    write('comparison.json',dict(status='PASS',old_case=old,cases=summary,pixel_pitch_um=10,
        note='Post-hoc illustrative selection, not a performance estimate. Marker positions, focus and contrast are generated from saved measurements without truth-based adjustment. Current paper figure is not yet replaced.'))
    document='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Fig.9(a) · 位置精度を確認した別事例</title><style>
body{margin:0;font:16px/1.65 system-ui,sans-serif;background:#f6f7f8;color:#17242d}main{max-width:1680px;padding:24px;margin:auto}h1{font-size:clamp(24px,4vw,36px)}h2{font-size:24px}a{color:#155d8a;text-underline-offset:3px}section{background:white;border:1px solid #dce2e6;border-radius:12px;padding:20px;margin:28px 0}.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figure img{width:100%;height:auto;display:block;image-rendering:pixelated}figcaption{font-variant-numeric:tabular-nums;white-space:nowrap}figure p{font-size:13px;color:#53616b;margin:4px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #dce2e6;white-space:nowrap}.note{color:#52616b;max-width:90ch}.red{color:#D84A3A}.blue{color:#2C6FB7}@media(max-width:1000px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){main{padding:14px}section{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 12px}}
</style></head><body><main><a href="fig09-frames.html#b">(b)の比較画像へ</a><h1>Fig.9(a)：中心位置を確認した別事例</h1>
<p><strong>前の(a)は、中心検出に失敗した例を選んでいました。</strong>赤い軌跡の元計測は、正解中心から12.27〜12.67画素（123〜127 µm）離れていました。平滑化後も12.47〜14.10画素（125〜141 µm）の誤差が残っています。</p>
<p class="note">元の検出枠は粒子の輪郭付近にあり、幅約7〜8画素・高さ約8〜12画素でした。画像から求めた重心が検出枠の中心から2画素を超えて離れた際に、枠中心へ戻す処理が4コマとも適用されています。したがって、表示上のずれではなく保存された計測値自体のずれです。粒径と「正解粒子の投影円内」という条件だけで選んだのが不適切でした。</p>
<p>下は、親2粒子・娘粒子との対応、全6コマの位置誤差、実画像を確認して選び直した候補です。<strong>C1は大きさを保つ候補、C3は位置精度を優先する候補</strong>です。C1の合体直後の青点には4.43画素の誤差が残っています。</p>
<div class="table"><table><thead><tr><th>事例</th><th>真値径 (µm)</th><th>平均XY誤差 (px)</th><th>最大XY誤差 (px)</th><th>補足</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p class="note">誤差は表示する6コマの実観測10点（親2点×4コマ＋娘1点×2コマ）について、平滑化後の計測中心と正解中心の距離を集計したものです。1画素＝10 µm。テスト系列全体の性能値ではありません。</p>
<p><span class="red">赤丸：終了軌跡</span>、<span class="blue">青四角：継続軌跡</span>。線は直前1 ms。クロップは対象中心の128画素、焦点は継続軌跡の推定Zです。点の位置を真値へ移したり、真値で焦点を合わせたりしていません。撮影範囲外だけ黒表示です。</p>
<p><a href="fig09-clean-a/candidates.zip" download>軌跡付き24枚と横並び画像を保存</a> · <a href="fig09-clean-a/comparison.json" download>位置誤差と原因の記録</a></p>
'''+''.join(sections)+'''
<p>使用する画像は「C1-1、C1-4、C1-6」のように指定できます。(b)は変更していません。現在のFig.9 PDFの(a)はまだ選び直し前の例です。このページは差し替え候補の確認用です。</p>
<footer class="note">既存のQuick Tunnelを使用。プロセス停止で閲覧できなくなり、再作成するとURLが変わります。</footer></main></body></html>'''
    (OUT/'fig09-clean-a.html').write_text(document)
    files=[]
    for c in cases:
        cid=c['case_id']
        for p in sorted((OUT/cid/'tracks').glob('*.png')):
            target=PUBLIC/'fig09-clean-a'/cid/p.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);files.append(target)
        target=PUBLIC/'fig09-clean-a'/f'{cid}-frames.png';shutil.copyfile(OUT/f'{cid}-frames.png',target);files.append(target)
    with zipfile.ZipFile(OUT/'candidates.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(PUBLIC/'fig09-clean-a')))
    for name in ['candidates.zip','comparison.json']:
        target=PUBLIC/'fig09-clean-a'/name;shutil.copyfile(OUT/name,target);files.append(target)
    target=PUBLIC/'fig09-clean-a.html';shutil.copyfile(OUT/'fig09-clean-a.html',target);files.append(target)
    # Keep the old gallery's (b) and labels intact, with a direct route to the corrected (a) candidates.
    old_page=PUBLIC/'fig09-frames.html'
    text=old_page.read_text()
    if 'id="a-correction"' not in text:
        text=text.replace('<h1>Fig.9の表示フレームを選ぶ</h1>',
            '<h1>Fig.9の表示フレームを選ぶ</h1><p id="a-correction"><strong>下の(a)は中心位置のずれが大きいため、事例を選び直しています。</strong><a href="fig09-clean-a.html">位置精度を確認した新しい(a)の候補を見る</a>。(b)はそのままです。</p>')
        old_page.write_text(text)
    assert 'id="a-correction"' in text and 'href="fig09-clean-a.html"' in text
    files.append(old_page)
    receipt=json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)):sha(p) for p in files})
    receipt['total_bytes']=sum((PUBLIC/name).stat().st_size for name in receipt['files'])
    temp=PREVIEW/'public-manifest.clean-a.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    assert protected=={path:sha(path) for path in protected}
    write('verification.json',dict(status='PASS',protected_paper_outputs=protected,old_observations=len(old['points']),
        new_cases=len(cases),new_observations=sum(len(c['source']['observations']) for c in cases),published_paths=[str(p.relative_to(PUBLIC)) for p in files]))
    print(json.dumps({'status':'PASS','new_cases':len(cases),'new_observations':40,'published_files':len(files),'old_mean_px':old['all_points_mean_error_px'],'old_max_px':old['all_points_max_error_px']},indent=2))


if __name__=='__main__':main()
