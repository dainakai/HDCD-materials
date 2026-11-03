#!/usr/bin/env python3
"""Publish checked same-particle examples; never substitute distinct particles."""
import hashlib
import json
import re
import shutil
import zipfile

from shared_particle import OUT, REPO, ROOT, KEY, write
from frame_gallery import PUBLIC, PREVIEW, sha, time_label
from confirm_true import OUT as CONFIRMED
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CHOICES={'D4':[-2,0,2],'D2':[-1,0,1]}
NOTES={
    'D4':dict(title='D4 · 第一候補：321 µm、最終候補からも除外',
        text='赤ID 17945と青ID 18031が、同じ真値粒子7364を間欠的に追跡しています。−0.50 msと0 msでは両方がその1個に対応し、＋0.50 msでは赤が終了して青だけが残ります。',
        caveat='掲載候補は−0.50・0・＋0.50 msです。−0.75・−0.25 msでは赤の位置が対象粒子から外れ、＋0.25 msでは青が欠測しています。これらは掲載候補から外し、下の診断欄に表示しています。候補時刻とスコアはframe 5337の保存値のままです。',
        history='粒子7364はframe 5312の合体で生じた娘粒子です（候補の6.25 ms前）。青ID 18031はその後のframe 5319に始まるため、この赤青ペアが以前の2つの親粒子を表すわけではありません。'),
    'D2':dict(title='D2 · 比較候補：319 µm、より大きいスコア低下',
        text='赤ID 18567と青ID 18658が、同じ真値粒子7645を追跡しています。−0.25 msと0 msでは両方が1個の粒子上にあり、＋0.25 msでは赤が終了して青だけが残ります。',
        caveat='スコアは0.937→0.488に低下しますが、最終候補一覧には残っています。採用する場合、元図の「rejected」は使えません。−1.00 msの青点は真値粒子の外周より約1.38 px外にあるため、掲載候補にはしていません。',
        history='2本はframe 5500の真の合体以前には別々の親粒子を追い、その後は娘粒子7645を重複追跡しています。表示区間は合体後で、候補時刻は真の合体の2.00 ms後です。D4よりも解釈上の補足が必要です。'),
}


def triptych(cid, panels, score):
    size,gap,margin,top=256,16,16,110
    sheet=Image.new('RGB',(2*margin+3*size+2*gap,top+size+34),'white')
    draw=ImageDraw.Draw(sheet)
    font='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    title=ImageFont.truetype(font,25);label=ImageFont.truetype(font,22);small=ImageFont.truetype(font,18)
    draw.text((margin,8),f'{cid}: one true particle / {score[0]:.3f} -> {score[1]:.3f}',font=title,fill='#17242d')
    draw.ellipse((margin,47,margin+13,60),fill='#D84A3A')
    draw.text((margin+23,40),'terminating track',font=label,fill='#D84A3A')
    draw.rectangle((300,47,313,60),fill='#2C6FB7')
    draw.text((323,40),'continuing track',font=label,fill='#2C6FB7')
    for i,p in enumerate(panels):
        x=margin+i*(size+gap)
        draw.text((x,76),f'{cid}  {time_label(p["delta"])}',font=label,fill='#17242d')
        with Image.open(OUT/cid/'tracks'/p['filename']) as im:
            assert im.size==(256,256);sheet.paste(im,(x,top))
            assert np.array_equal(np.asarray(sheet)[top:top+size,x:x+size],np.asarray(im))
        draw.text((x,top+size+6),f'frame {p["frame"]}',font=small,fill='#53616b')
    sheet.save(OUT/f'{cid}-proposed-three.png')


def main():
    final=REPO.parent/'audit/revision-figure9-final-20261103/selection.json'
    if final.exists() and json.loads(final.read_text()).get('author_confirmed'):
        raise RuntimeError('N1/D4 are now author-confirmed. Use finalize.py and publish_final.py; do not restore the pre-selection candidate page.')
    a=json.loads((CONFIRMED/'selection.json').read_text())
    assert a['status']=='author_confirmed' and a['display_frames']==[3957,3958,3959]
    protected=a['old_complete_figure_and_zip_unchanged']
    protected.update({str(CONFIRMED/'fig09-a-confirmed.pdf'):a['pdf_sha256'],
                      str(CONFIRMED/'fig09-a-confirmed.png'):a['png_sha256']})
    assert all(sha(p)==expected for p,expected in protected.items())
    paper=REPO.parent/'CollisionPaper/本番フォーマット/template.tex'
    assert sha(paper)=='1fbbcfcf35f5011067d92429359fd7785f98653ace260b49622b32f60e1e83b5'
    manifest=json.loads((REPO/'experiments/amt_full_20261101/results/manifest.json').read_text())['numerical_outputs_manifest']
    assert all(sha(ROOT/p)==expected for p,expected in manifest.items())
    feat=pd.read_parquet(ROOT/'data/features/validation/seed260745/paper92.parquet').set_index(KEY)
    scores=pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet').set_index(KEY)
    pred=pd.read_parquet(ROOT/'results/ours/predictions.parquet')
    final=set(zip(pred.vanished_track_id,pred.survivor_track_id,pred.candidate_frame_raw))
    cases=json.loads((OUT/'rendered-cases.json').read_text())
    diagnostics=json.loads((OUT/'diagnostic-frames.json').read_text())
    published=[];zip_members=[];sections=[];summaries=[];table=[]

    def publish(p, relative):
        dest=PUBLIC/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest);published.append(dest)
        return dest

    def card(cid,p,diagnostic=False):
        base=f'fig09-shared-particle/{cid}'
        links=f'<a href="{base}/images/{p["filename"]}">画像のみ</a>'
        if not diagnostic:links+=f' · <a href="{base}/proof/{p["filename"]}">真値の輪郭</a>'
        badge=' · 掲載対象外' if diagnostic else ''
        return f'<figure><figcaption>{time_label(p["delta"])}{badge}</figcaption><a href="{base}/tracks/{p["filename"]}"><img src="{base}/tracks/{p["filename"]}" width="256" height="256" alt="{cid} {time_label(p["delta"])}、赤丸・青四角と実観測の軌跡"></a><p>frame {p["frame"]} · {links}</p></figure>'

    for cid,deltas in CHOICES.items():
        c=next(c for c in cases if c['case_id']==cid);r=c['source']
        key=tuple(int(r[k]) for k in KEY)
        assert float(feat.loc[key].static_score)==r['static_score']>=.9
        assert int(feat.loc[key].image_enriched)==1 and int(feat.loc[key].rendered_pre_frames)==9
        assert float(scores.loc[key].base_score)==r['base_score']<r['static_score']
        assert ((key in final)==r['final_retained']) and r['no_truth_event_in_common_gate']
        selected=[next(p for p in c['panels'] if p['delta']==d) for d in deltas]
        observations=[]
        for p in c['panels']:
            proof=p['truth_audit']
            assert p['recorded_fraction']==1 and proof['whole_crop']
            for point in [proof['blue_observation'],proof['red_observation']]:
                if point is None:continue
                assert point['truth_id']==r['shared_truth_id'] and point['plausible_truth_ids']==[r['shared_truth_id']]
                assert point['nearest_xy_truth_id']==r['shared_truth_id'] and point['mapping_agrees']
                assert point['xy_error_px']<=point['truth_diameter_um']/20 and abs(point['z_error_um'])<100
                observations.append(point)
            for kind in ['images','tracks','proof']:
                source=OUT/cid/kind/p['filename'];dest=publish(source,f'fig09-shared-particle/{cid}/{kind}/{source.name}')
                zip_members.append((dest,f'{cid}/{kind}/{source.name}'))
        assert selected[0]['truth_audit']['red_same'] and selected[1]['truth_audit']['red_same']
        assert not selected[2]['truth_audit']['red_present']
        triptych(cid,selected,(r['static_score'],r['base_score']))
        for name in [f'{cid}-frames.png',f'{cid}-proposed-three.png']:
            dest=publish(OUT/name,f'fig09-shared-particle/{name}');zip_members.append((dest,name))
        diag=[p for p in diagnostics if p['case_id']==cid]
        for p in diag:
            for kind in ['images','tracks']:
                source=OUT/cid/kind/p['filename'];dest=publish(source,f'fig09-shared-particle/{cid}/{kind}/{source.name}')
                zip_members.append((dest,f'{cid}/{kind}/{source.name}'))
        summ=dict(case_id=cid,status='reviewed_candidate_not_author_selected',candidate_key=list(key),
            source_particle_id=r['shared_truth_id'],truth_diameter_um=observations[0]['truth_diameter_um'],
            trajectory_score=r['static_score'],image_feature_score=r['base_score'],score_decrease=r['score_decrease'],
            shared_component_count=int(r['pre_same_interp_count']),rendered_pre_frames=int(r['rendered_pre_frames']),
            final_retained=r['final_retained'],no_truth_event_in_common_gate=True,
            selected_frames=[p['frame'] for p in selected],selected_times_ms=[d/4 for d in deltas],
            minimum_selected_crop_hologram_margin_px=min(p['hologram_margin_px'] for p in selected),
            maximum_displayed_z_error_um=max(abs(p['z_error_um']) for p in observations),
            observations=observations,selected_panels=selected,diagnostics=diag,
            interpretation=NOTES[cid],related_truth_events=r['related_truth_events'])
        summaries.append(summ)
        retained='残る（rejected不可）' if r['final_retained'] else '除外される'
        table.append(f'<tr><td><a href="#{cid}">{cid}</a></td><td>{summ["truth_diameter_um"]:.0f} µm</td><td>{r["static_score"]:.6f} → {r["base_score"]:.6f}</td><td>−{r["score_decrease"]:.6f}</td><td>{retained}</td></tr>')
        diag_note=('＋0.25 msは青の実観測がないため、診断画像の中心とZだけを前後の実観測から線形補間しています。青点・青線は描いていません。' if cid=='D4' else '−1.00 msも撮影範囲内ですが、青点が真値粒子の投影円の外側にあります。')
        sections.append(f'''<section id="{cid}"><h2>{NOTES[cid]['title']}</h2><p>{NOTES[cid]['text']}</p>
<div class="frames three">{''.join(card(cid,p) for p in selected)}</div>
<p><strong>3枚の候補：</strong>{'、'.join(time_label(d) for d in deltas)}。128画素クロップの外縁から撮影境界まで、最短でも{summ['minimum_selected_crop_hologram_margin_px']} pxあります。</p>
<p class="caveat">{NOTES[cid]['caveat']}</p><p>{NOTES[cid]['history']}</p>
<details><summary>他の確認済みフレームを見る</summary><div class="frames">{''.join(card(cid,p) for p in c['panels'])}</div></details>
<details><summary>掲載対象外のフレーム・欠測を確認する</summary><p>{diag_note}</p><div class="frames three">{''.join(card(cid,p,True) for p in diag)}</div></details>
<p><a href="fig09-shared-particle/{cid}-proposed-three.png">3枚を横並びで開く</a></p></section>''')
    summary=dict(status='PASS',author_selected_b=False,confirmed_a='N1: -0.25/0/+0.25 ms',
        withdrawn_examples=['B01','B02'],paper_definition='Two observed track IDs intermittently track the same physical particle; not two distinct true particles with overlapping projections.',
        cases=summaries,phase_iterations=3,crop_size_px=128,pixel_pitch_um=10,
        contrast='Original per-frame recorded-pixel 1st/99th percentile normalization.',
        marker_policy='Saved smoothed track observations and recent 1 ms tails; no truth-snapping or fabricated marker observations.',
        search=json.loads((OUT/'search-protocol.json').read_text()),
        intermittent_search=json.loads((OUT/'intermittent-search-protocol.json').read_text()),
        scope='Post-hoc illustrative review only. Saved models, candidate scores, 34 numerical outputs and confirmed (a) are unchanged.')
    write('review.json',summary)
    dest=publish(OUT/'review.json','fig09-shared-particle/review.json');zip_members.append((dest,'review.json'))
    for name in ['fig09-a-confirmed.pdf','fig09-a-confirmed.png','selection.json']:
        zip_members.append((CONFIRMED/name,'confirmed-a/'+name))
    note='These are reviewed Fig.9(b) candidates, not an author selection. B01/B02 are withdrawn. D4 shows one true particle with two track IDs, scores 0.943667 -> 0.624751, and is absent from final predictions. D2 shows duplicate tracking after a true collision, scores 0.937295 -> 0.488098, but remains in final predictions: do not label it rejected. See review.json for times, independent truth checks and diagnostic exceptions. Confirmed (a) N1 is unchanged.\n'
    (OUT/'README.txt').write_text(note)
    zip_members.append((OUT/'README.txt','README.txt'))
    with zipfile.ZipFile(OUT/'review-assets.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p,name in zip_members:z.write(p,name)
    publish(OUT/'review-assets.zip','fig09-shared-particle/review-assets.zip')
    # The old URL remains a correction notice, including direct old downloads.
    page='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Fig.9(b) · 同じ真の粒子を追う2本の軌跡</title><style>
body{margin:0;background:#f6f7f8;color:#17242d;font:16px/1.75 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:clamp(24px,4vw,36px)}h2{font-size:24px}a{color:#155d8a;text-underline-offset:3px}section{background:white;border:1px solid #dce2e6;border-radius:12px;padding:20px;margin:26px 0}.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:16px}.three{grid-template-columns:repeat(3,minmax(0,1fr));max-width:920px}figure{margin:0;min-width:0}figure img{width:100%;height:auto;display:block;image-rendering:pixelated}figcaption{font-weight:600;margin-bottom:6px}figure p{font-size:13px;color:#53616b;margin:4px 0}.confirmed{width:min(100%,1000px);height:auto}.note{color:#52616b}.caveat{padding:12px 16px;background:#fff8e6;border-left:4px solid #c48a16}.correction{padding:12px 16px;background:#fcecec;border-left:4px solid #ac4444}.table{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px 12px;border-bottom:1px solid #dce2e6;text-align:left;white-space:nowrap}details{margin:18px 0;padding:10px 0}summary{cursor:pointer;font-weight:600}details[open] summary{margin-bottom:16px}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:1000px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){main{padding:14px}section{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 12px}.three{grid-template-columns:1fr;max-width:340px}figcaption{font-size:15px}}</style></head><body><main>
<a href="synthetic.html#fig9">Fig.9と評価結果へ</a><h1>Fig.9(b)：同じ真の粒子を追う2本の軌跡</h1>
<p class="correction"><strong>前回のB01・B02は撤回しました。</strong>別々の真値粒子を追う例で、論文の(b)の意味に合っていませんでした。キャプションを別の現象に書き換える案も撤回しています。</p>
<p>CollisionPaperの本文とFig.9の説明を再確認しました。必要なのは、<strong>1個の物理粒子に2つの観測軌跡IDが割り当てられる重複追跡</strong>です。以下は真値のID・XY位置・奥行きとの照合と、再構成画像の目視確認を終えた候補です。</p>
<p>第一候補は<strong>D4（321 µm、0.944→0.625）</strong>です。元図と同じ現象を示し、最終候補からも除外されています。より大きいスコア低下のD2も比較できますが、こちらには後述の注意点があります。</p>
<div class="table"><table><thead><tr><th>候補</th><th>真値粒径</th><th>trajectory → ＋image features</th><th>低下量</th><th>最終候補一覧</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>
<p>赤丸・赤線は終了軌跡、青四角・青線は継続軌跡です。点は保存された観測座標、線は直前1 msの軌跡です。各画像の「真値の輪郭」では黄色の円と十字で真値の粒径と中心を確認できます。黄色は検証用で、論文用画像には含めません。</p>
<div class="links"><a href="fig09-shared-particle/review-assets.zip" download>確認画像・照合記録を保存</a><a href="fig09-shared-particle/review.json">真値とスコアの照合記録</a></div>
'''+''.join(sections)+'''
<section id="a"><h2>(a)はN1の確定内容を維持</h2><p>−0.25・0・＋0.25 ms（3957・3958・3959）。スコア0.998955→0.997927。</p><img class="confirmed" src="fig09-confirmed-a/fig09-a-confirmed.png" alt="確定済みN1の3枚"><p><a href="fig09-confirmed-a/fig09-a-confirmed.pdf">確定した(a)単独PDF</a></p></section>
<section><h2>確認した範囲</h2><p>保存済み263,185候補から、trajectory scoreが0.9以上かつ画像特徴追加後にスコアが下がる13,031件を確認しました。2つの軌跡が同じ真値粒子に対応するかを独立に調べ、観測が間欠的な例も追加で確認しています。再構成画像の共有成分だけで同一粒子とは判定していません。</p><p>掲載候補の全点は同じ真値粒子の投影円内にあり、奥行き誤差は100 µm未満です。全クロップ128×128画素が撮影範囲内に収まっています。位相回復は3回、焦点は継続軌跡の推定Z、コントラスト・点・線は元の作図処理と同じです。診断欄の例外は個別に明記しています。</p><p>確定済み(a)、学習済みモデル、保存済みスコア、34件の固定した評価出力は変更していません。(b)はまだ選定段階で、Fig.9全体と論文差し替えZIPの更新前です。</p></section>
<footer class="note">既存Quick Tunnelを使用。プロセス停止で閲覧できなくなり、再作成するとURLが変わります。</footer></main></body></html>'''
    (OUT/'fig09-artifact-options.html').write_text(page)
    publish(OUT/'fig09-artifact-options.html','fig09-artifact-options.html')
    notice='<p id="fig9-current-selection"><strong>現在の選択：</strong>(a)はN1の−0.25・0・＋0.25 msで確定。旧B01・B02は撤回しました。<a href="fig09-artifact-options.html">同じ真値粒子を追う(b)の確認済み候補</a>をご覧ください。この下の旧事例・全体PDFは比較用です。</p>'
    for name in ['fig09-interior-a.html','fig09-clean-a.html','fig09-frames.html','synthetic.html']:
        p=PUBLIC/name;text=p.read_text();text,n=re.subn(r'<p id="fig9-current-selection">.*?</p>',notice,text,count=1,flags=re.S)
        assert n==1;p.write_text(text);published.append(p)
    receipt=json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)):sha(p) for p in published})
    receipt['total_bytes']=sum((PUBLIC/p).stat().st_size for p in receipt['files'])
    temp=PREVIEW/'public-manifest.shared-particle.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    assert all(sha(p)==expected for p,expected in protected.items())
    assert all(sha(ROOT/p)==expected for p,expected in manifest.items())
    write('publication-inputs.json',dict(status='PASS',published_paths=[str(p.relative_to(PUBLIC)) for p in published],
        zip_members={name:sha(p) for p,name in zip_members},protected_files=protected,
        manuscript_sha256=sha(paper),numerical_outputs_unchanged=len(manifest)))
    target=REPO/'experiments/amt_figure9_refinement_20261102/results/shared-particle';target.mkdir(exist_ok=True)
    for name in ['review.json','D4-proposed-three.png','D2-proposed-three.png']:
        shutil.copyfile(OUT/name,target/name)
    print(json.dumps({'status':'PASS','reviewed_candidates':list(CHOICES),'published_files':len(published),'zip_entries':len(zip_members)},indent=2))


if __name__=='__main__':main()
