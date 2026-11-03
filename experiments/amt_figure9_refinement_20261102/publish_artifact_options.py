#!/usr/bin/env python3
"""Publish confirmed (a), reviewed (b) alternatives, and their distinct meanings."""
import json
import shutil
import zipfile

from interior_artifact import OUT, REPO, write
from confirm_true import OUT as CONFIRMED
from frame_gallery import PREVIEW, PUBLIC, FINAL, sha, time_label

DESCRIPTIONS={
    'B01':'別々の粒子の投影像が重なる例。真値の奥行き差は約4.68 mmあり、衝突していません。画像特徴は衝突前9コマ全てで共有成分を示しますが、真値では同一粒子ではありません。',
    'B02':'異なる奥行きの別粒子を誤って組み合わせた例。真値の奥行き差は約7.84〜7.87 mmで、表示中の投影像も分かれています。スコア低下はB01より小さめです。',
}


def main():
    a=json.loads((CONFIRMED/'selection.json').read_text())
    assert a['status']=='author_confirmed' and a['display_frames']==[3957,3958,3959]
    protected=a['old_complete_figure_and_zip_unchanged']
    assert protected=={p:sha(p) for p in protected}
    cases=json.loads((OUT/'rendered-cases.json').read_text())
    assert len(cases)==2
    files=[];zip_files=[];rows=[];sections=[];summaries=[]
    for name in ['fig09-a-confirmed.pdf','fig09-a-confirmed.png','selection.json']:
        dest=PUBLIC/'fig09-confirmed-a'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(CONFIRMED/name,dest);files.append(dest);zip_files.append((dest,'confirmed-a/'+name))
    for kind in ['images','tracks']:
        for p in sorted((CONFIRMED/kind).glob('*.png')):
            dest=PUBLIC/'fig09-confirmed-a'/kind/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);files.append(dest);zip_files.append((dest,'confirmed-a/'+kind+'/'+p.name))
    for c in cases:
        cid,r=c['case_id'],c['source'];panels=c['panels']
        assert all(p['recorded_fraction']==1 for p in panels)
        rows.append(f'<tr><td><a href="#{cid}">{cid}</a></td><td>{r["static_score"]:.6f}</td><td>{r["base_score"]:.6f}</td><td>−{r["score_decrease"]:.6f}</td><td>{r["min_hologram_margin_px"]} px</td></tr>')
        cards=[]
        for p in panels:
            over=f'fig09-artifact-options/{cid}/tracks/{p["filename"]}'
            plain=f'fig09-artifact-options/{cid}/images/{p["filename"]}'
            cards.append(f'<figure><figcaption><strong>{p["label"]}</strong> {time_label(p["delta"])}</figcaption><a href="{over}"><img src="{over}" width="256" height="256" alt="{p["label"]}、赤青の実測軌跡付き、ホログラム内のクロップ"></a><p>frame {p["frame"]} · <a href="{plain}">画像のみ</a></p></figure>')
            for kind in ['images','tracks']:
                dest=PUBLIC/'fig09-artifact-options'/cid/kind/p['filename'];dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(OUT/cid/kind/p['filename'],dest);files.append(dest);zip_files.append((dest,'b-candidates/'+cid+'/'+kind+'/'+p['filename']))
        dest=PUBLIC/'fig09-artifact-options'/f'{cid}-frames.png'
        shutil.copyfile(OUT/f'{cid}-frames.png',dest);files.append(dest);zip_files.append((dest,'b-candidates/'+dest.name))
        labels=[p['label'] for p in panels if p['delta'] in [-1,0,1]]
        note=('−1.00 msには継続軌跡の実観測がないため、この一覧は−0.75〜＋0.50 msです。補間した点は描いていません。' if r['missing_minus1ms_observation'] else '')
        sections.append(f'''<section id="{cid}"><h2>{cid}{' · 第一候補' if cid=='B01' else ' · 比較候補'}</h2><p>{DESCRIPTIONS[cid]}</p>
<div class="frames">{''.join(cards)}</div><p class="note">{note} (a)と同じ−0.25、0、＋0.25 msなら、{ '・'.join(labels) }の3枚です。</p><p><a href="fig09-artifact-options/{cid}-frames.png">6枚を横一列で開く</a></p></section>''')
        summaries.append(dict(case_id=cid,kind=r['kind'],candidate_frame=int(r['candidate_frame']),
            vanished_track_id=int(r['vanished_track_id']),survivor_track_id=int(r['survivor_track_id']),
            trajectory_score=r['static_score'],image_feature_score=r['base_score'],score_decrease=r['score_decrease'],
            image_shared_component_fraction=r['pre_same_interp_fraction'],truth_ids_by_track=r['truth_ids_by_track'],
            truth_z_separation_um_range=r['truth_z_separation_um_range'],truth_projected_edge_gap_px_range=r['truth_projected_edge_gap_px_range'],
            min_hologram_margin_px=r['min_hologram_margin_px'],display_deltas=r['display_deltas'],panels=r['panels'],
            observations=r['observations'],nearest_event_in_time_gate=r['nearest_event_in_time_gate'],
            final_candidate_retained=False,description=DESCRIPTIONS[cid]))
    write('comparison.json',dict(status='PASS',a_status='author_confirmed',a_case='N1',a_times_ms=[-.25,0,.25],
        b_status='candidates_only_not_author_selected',cases=summaries,
        scientific_note='Both b candidates represent distinct true particles. Do not call them two track IDs on one true particle. No scheduled collision exists between either pair in the test series; neither candidate matches a true event in the common evaluation gate.',
        search_protocol=json.loads((OUT/'search-protocol.json').read_text())))
    caption='''Confirmed (a): Ground-truth collision, 154 + 242 -> 261 micrometres. Display times: -0.25, 0, +0.25 ms. Trajectory score 0.998955; after image features 0.997927.

If B01 is selected: A spurious candidate formed by two distinct particles whose images overlap in projection. Their true axial separation is approximately 4.68 mm. The trajectory score decreases from 0.917328 to 0.489228 after adding image features. The two track IDs do not refer to the same true particle.

If B02 is selected: A spurious pairing of distinct particles at different depths, with a true axial separation of approximately 7.84-7.87 mm. Their projected images are separated in the displayed frames. The trajectory score decreases from 0.918814 to 0.656951 after adding image features.

Replace the previous panel title "Shared-particle artifact" with an accurate title such as "Spurious particle-pair candidate" if either new candidate is selected. The previous caption's same-true-particle interpretation must not be retained. These descriptions are proposals, not edits to the manuscript.
'''
    (OUT/'caption-options.txt').write_text(caption)
    for name in ['comparison.json','caption-options.txt']:
        dest=PUBLIC/'fig09-artifact-options'/name;shutil.copyfile(OUT/name,dest);files.append(dest);zip_files.append((dest,name))
    with zipfile.ZipFile(OUT/'review-assets.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p,name in zip_files:z.write(p,name)
    dest=PUBLIC/'fig09-artifact-options/review-assets.zip';shutil.copyfile(OUT/'review-assets.zip',dest);files.append(dest)
    page='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Fig.9 · (a)確定と(b)の境界内候補</title><style>
body{margin:0;background:#f6f7f8;color:#17242d;font:16px/1.65 system-ui,sans-serif}main{max-width:1680px;margin:auto;padding:24px}h1{font-size:clamp(24px,4vw,36px)}h2{font-size:24px}a{color:#155d8a;text-underline-offset:3px}section{background:white;border:1px solid #dce2e6;border-radius:12px;padding:20px;margin:26px 0}.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figure img{width:100%;height:auto;display:block;image-rendering:pixelated}figcaption{white-space:nowrap}figure p{font-size:13px;color:#53616b;margin:4px 0}.confirmed{width:min(100%,1100px);height:auto}.note{color:#52616b;max-width:100ch}.table{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px 12px;border-bottom:1px solid #dce2e6;text-align:left;white-space:nowrap}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:1000px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){main{padding:14px}section{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 12px}figcaption{font-size:14px}}</style></head><body><main>
<a href="synthetic.html#fig9">Fig.9と結果のページへ</a><h1>Fig.9：(a)確定と、境界で見切れない(b)候補</h1>
<section id="a"><h2>(a)確定 · N1</h2><p><strong>−0.25、0、＋0.25 ms（frame 3957・3958・3959）</strong>の3枚で固定しました。真値粒径154＋242→261 µm、スコア0.998955→0.997927です。</p>
<img class="confirmed" src="fig09-confirmed-a/fig09-a-confirmed.png" alt="確定したN1の3コマ、元の図と同じ軌跡・スケールバー・時刻ラベル">
<p><a href="fig09-confirmed-a/fig09-a-confirmed.pdf" download>(a)単独PDF</a> · <a href="fig09-confirmed-a/selection.json" download>確定した事例とフレームの記録</a></p></section>
<p><strong>(b)の第一候補はB01です。</strong>trajectory score 0.917から、画像特徴の追加で0.489に下がります。B02も比較用に示します。両方とも128画素クロップ全体がホログラム内に収まり、全12コマに黒い余白や対象粒子の見切れはありません。</p>
<p><strong>前の(b)とはアーティファクトの種類が異なります。</strong>今回はどちらも真値では別々の粒子です。B01は投影像の重なり、B02は離れた粒子の誤った組合せを示します。「同じ真値粒子に2本の軌跡」という従来の説明は使用できないため、採用時はパネル名・キャプションも対応する記述へ変えます。</p>
<div class="table"><table><thead><tr><th>候補</th><th>trajectory score</th><th>画像特徴追加後</th><th>変化</th><th>境界までの余裕</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p class="note">境界までの余裕は、全6コマのクロップ外縁からホログラム境界までの最短距離です。両候補とも最終出力から除外され、共通の評価条件内に対応する衝突はありません。真値を追跡し、後で実際に衝突する粒子対を早く検出しただけの例は、この候補一覧から除いています。</p>
<p>赤丸・赤線と青四角・青線は保存された計測座標と直前1 msの軌跡です。位相回復3回、128画素、継続軌跡の推定Z、元の図と同じコントラスト調整で再構成しています。</p>
<div class="links"><a href="fig09-artifact-options/review-assets.zip" download>(a)確定版と(b)候補画像を保存</a><a href="fig09-artifact-options/caption-options.txt">キャプション修正案</a><a href="fig09-artifact-options/comparison.json">真値・スコア・位置の照合記録</a></div>
'''+''.join(sections)+'''
<p>完成版Fig.9全体と論文差し替えZIPの更新は(b)決定後に行います。このページの(a)単独PDFは確定した3コマです。既存の全体PDFは旧版として保持しています。性能評価値・モデル・原稿本文は変更していません。</p>
<footer class="note">既存Quick Tunnelを使用。プロセス停止で閲覧できなくなり、再作成するとURLが変わります。</footer></main></body></html>'''
    (OUT/'fig09-artifact-options.html').write_text(page)
    dest=PUBLIC/'fig09-artifact-options.html';shutil.copyfile(OUT/'fig09-artifact-options.html',dest);files.append(dest)
    notice='<p id="fig9-current-selection"><strong>現在の選択：</strong>(a)はN1の−0.25・0・＋0.25 msで確定しました。<a href="fig09-artifact-options.html">確定した(a)と、境界で見切れない(b)候補を見る</a>。この下の旧事例・全体PDFは比較用です。</p>'
    for name in ['fig09-interior-a.html','fig09-clean-a.html','fig09-frames.html','synthetic.html']:
        p=PUBLIC/name;text=p.read_text()
        if 'id="fig9-current-selection"' not in text:
            if name=='synthetic.html':
                start=text.index('id="fig9"');at=text.index('>',start)+1
            else:at=text.index('</h1>')+len('</h1>')
            text=text[:at]+notice+text[at:];p.write_text(text)
        files.append(p)
    receipt=json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)):sha(p) for p in files})
    receipt['total_bytes']=sum((PUBLIC/n).stat().st_size for n in receipt['files'])
    temp=PREVIEW/'public-manifest.artifact-options.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    assert protected=={p:sha(p) for p in protected}
    write('publication-inputs.json',dict(status='PASS',published_paths=[str(p.relative_to(PUBLIC)) for p in files],
        zip_members={name:sha(p) for p,name in zip_files},a_confirmed=True,b_selected=False,
        protected_complete_figure_and_zip=protected))
    print(json.dumps({'status':'PASS','a':'N1 at -0.25/0/+0.25ms','b_candidates':2,'files':len(files),'zip_entries':len(zip_files)},indent=2))


if __name__=='__main__':main()
