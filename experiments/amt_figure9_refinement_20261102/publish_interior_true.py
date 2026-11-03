#!/usr/bin/env python3
"""Publish the author-requested boundary-free Fig.9(a) candidate gallery."""
import json
import shutil
import zipfile

from interior_true import OUT, REPO, write
from frame_gallery import PREVIEW, PUBLIC, FINAL, sha, time_label

NOTES = {
    'N1': '大きさを重視する候補。＋0.25 msの青点には4.31画素の誤差が残ります。',
    'N2': '粒子の大きさと、対象付近の見やすさのバランスを取る候補。',
    'N3': '比較的大きな親粒子を保ち、全表示点の最大誤差は約2画素です。',
    'N4': '親2粒子の大きさが近い別例。対象の右側には背景粒子の像があります。',
    'N5': '境界から離れた候補。対象付近の背景像が比較的少なく、2粒子から1粒子への変化を見やすい例です。',
    'N6': '境界からさらに離れ、位置誤差も小さい候補。粒子の大きさは小さめです。',
}


def main():
    protected = {str(p): sha(p) for p in [FINAL/'fig09.pdf', FINAL/'selection.json',
        REPO.parent/'audit/revision-paper-replacements-20261102/paper-replacements.zip']}
    cases = json.loads((OUT/'rendered-cases.json').read_text())
    assert len(cases) == 6
    rows, sections, summary = [], [], []
    public_paths = []
    for c in cases:
        cid, r = c['case_id'], c['source']
        e = r['truth_event']
        assert len(c['panels']) == 6 and len(r['observations']) == 10
        assert all(p['recorded_fraction'] == 1 and p['hologram_margin_px'] >= 8 for p in c['panels'])
        assert min(p['hologram_margin_px'] for p in c['panels']) == r['min_hologram_margin_px']
        diameters = f"{e['parent1_diameter_um']:.0f}＋{e['parent2_diameter_um']:.0f}→{e['child_diameter_um']:.0f}"
        rows.append(f'<tr><td><a href="#{cid}">{cid}</a></td><td>{diameters}</td><td>{r["mean_xy_error_px"]:.2f} / {r["max_xy_error_px"]:.2f}</td><td>{r["min_hologram_margin_px"]}</td></tr>')
        cards = []
        for p in c['panels']:
            image_src = f'fig09-interior-a/{cid}/tracks/{p["filename"]}'
            plain_src = f'fig09-interior-a/{cid}/images/{p["filename"]}'
            cards.append(f'''<figure><figcaption><strong>{p['label']}</strong> {time_label(p['delta'])}</figcaption>
<a href="{image_src}"><img src="{image_src}" width="256" height="256" alt="{p['label']}：ホログラム内に全体が収まる128画素クロップ、赤丸・青四角の計測軌跡付き"></a>
<p>frame {p['frame']} · <a href="{image_src}" download>PNG</a> · <a href="{plain_src}">画像のみ</a></p></figure>''')
            for kind in ['tracks', 'images']:
                source = OUT/cid/kind/p['filename']
                target = PUBLIC/'fig09-interior-a'/cid/kind/p['filename']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                public_paths.append(target)
        target = PUBLIC/'fig09-interior-a'/f'{cid}-frames.png'
        shutil.copyfile(OUT/f'{cid}-frames.png', target)
        public_paths.append(target)
        maximum = max(r['observations'], key=lambda p: p['xy_error_px'])
        unmatched = [p for p in r['observations'] if not p['one_to_one_mapping_present']]
        correspondence_note = ''
        if unmatched:
            descriptions = [time_label(p['delta']) + 'の' + ('青点' if p['track_id'] == r['survivor_track_id'] else '赤点') for p in unmatched]
            correspondence_note = '<p class="note">' + '、'.join(descriptions) + 'は既存の一対一対応表では未対応です。今回、真値座標との距離と最も近い粒子IDを照合して対応を確認しています。</p>'
        summary.append(dict(case_id=cid, candidate_frame=int(r['candidate_frame']),
            event_id=int(e['collision_event_id']), vanished_track_id=int(r['vanished_track_id']),
            survivor_track_id=int(r['survivor_track_id']), parent_diameters_um=[e['parent1_diameter_um'],e['parent2_diameter_um']],
            child_diameter_um=e['child_diameter_um'], trajectory_score=r['static_score'], image_feature_score=r['base_score'],
            mean_xy_error_px=r['mean_xy_error_px'], max_xy_error_px=r['max_xy_error_px'],
            max_error_frame=maximum['frame'], max_error_track_id=maximum['track_id'],
            min_hologram_margin_px=r['min_hologram_margin_px'], crops=r['crops'],
            observations=r['observations'], note=NOTES[cid]))
        sections.append(f'''<section id="{cid}"><h2>{cid} · {diameters} µm</h2><p>{NOTES[cid]}</p>
<p class="note">XY位置誤差：平均 {r['mean_xy_error_px']:.2f}、最大 {r['max_xy_error_px']:.2f}画素。クロップ外縁からホログラム境界までの最短距離：{r['min_hologram_margin_px']}画素。trajectory score {r['static_score']:.6f} → image feature追加後 {r['base_score']:.6f}。</p>
<div class="frames">{''.join(cards)}</div>{correspondence_note}<p><a href="fig09-interior-a/{cid}-frames.png">6枚を横一列で開く</a></p></section>''')
    record = dict(status='PASS', cases=summary, crop_size_px=128, hologram_size_px=[1024,1024], pixel_pitch_um=10,
        phase_iterations=3, no_unrecorded_pixels_in_any_crop=True,
        focus='Actual continuing-track estimated Z; no truth-based refocusing.',
        note='Post-hoc illustrative candidates. No measurement coordinates, scores, fitted models or performance evaluation changed. Current Fig.9 PDF is unchanged pending author selection.')
    write('comparison.json', record)
    with zipfile.ZipFile(OUT/'candidates.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in public_paths:
            z.write(p, str(p.relative_to(PUBLIC/'fig09-interior-a')))
    document = '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Fig.9(a) · ホログラム境界で見切れない候補</title>
<style>body{margin:0;font:16px/1.65 system-ui,sans-serif;background:#f6f7f8;color:#17242d}main{max-width:1680px;padding:24px;margin:auto}h1{font-size:clamp(24px,4vw,36px)}h2{font-size:24px}a{color:#155d8a;text-underline-offset:3px}section{background:white;border:1px solid #dce2e6;border-radius:12px;padding:20px;margin:28px 0}.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figure img{width:100%;height:auto;display:block;image-rendering:pixelated}figcaption{font-variant-numeric:tabular-nums;white-space:nowrap}figure p{font-size:13px;color:#53616b;margin:4px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #dce2e6;white-space:nowrap}.note{color:#52616b;max-width:100ch}.red{color:#D84A3A}.blue{color:#2C6FB7}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:1000px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:560px){main{padding:14px}section{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 12px}figcaption{font-size:14px}}</style></head><body><main>
<a href="fig09-frames.html#b">(b)の比較画像へ</a><h1>Fig.9(a)：ホログラム境界で見切れない6候補</h1>
<p><strong>全36コマで128×128画素のクロップ全体がホログラム内に収まり、黒い余白はありません。</strong>対象粒子も全てクロップ内に収まっています。親2粒子・娘粒子との対応と全表示点の位置誤差を確認し、36コマの実画像を見て選びました。</p>
<p>見やすさと境界からの距離を重視するなら<strong>N5</strong>、より大きな粒子なら<strong>N1</strong>、粒径と位置精度のバランスなら<strong>N2・N3</strong>が候補です。N1の最大位置誤差は4.31画素なので、その点も比較してください。</p>
<div class="table"><table><thead><tr><th>候補</th><th>真値径：親2粒子→娘粒子 (µm)</th><th>XY誤差：平均 / 最大 (px)</th><th>クロップ外縁から境界まで (px)</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div>
<p class="note">位置誤差は6コマにある実観測10点（親2点×4コマ＋娘1点×2コマ）の集計です。テスト系列全体の性能値ではありません。境界までの距離は、全6コマ・上下左右のうちの最短距離です。ホログラムは1024×1024画素、1画素＝10 µmです。</p>
<p><span class="red">赤丸：終了軌跡</span>、<span class="blue">青四角：継続軌跡</span>。線は直前1 ms。時刻は−1.00、−0.75、−0.50、−0.25、0.00、＋0.25 msです。継続軌跡の実測位置を中心にクロップし、その推定Zで再構成しています。位相回復3回、元の図と同じコントラスト調整です。</p>
<p class="note">位置を真値に合わせる補正、クロップを内側にずらす操作、画像の埋め合わせはしていません。元の128画素画像は各コマの「画像のみ」から確認できます。</p>
<div class="links"><a href="fig09-interior-a/candidates.zip" download>候補画像一式を保存</a><a href="fig09-interior-a/comparison.json" download>位置・境界・スコアの記録</a><a href="fig09-clean-a.html">前回の候補と比較</a></div>
''' + ''.join(sections) + '''
<p>画像は「N5-1、N5-4、N5-6」のように指定できます。(b)と論文用Fig.9 PDFは変更していません。PDFの(a)はまだ旧例で、このページは差し替え候補の確認用です。</p>
<footer class="note">既存のQuick Tunnelを使用。プロセス停止で閲覧できなくなり、再作成するとURLが変わります。</footer></main></body></html>'''
    (OUT/'fig09-interior-a.html').write_text(document)
    for name in ['candidates.zip', 'comparison.json']:
        target = PUBLIC/'fig09-interior-a'/name
        shutil.copyfile(OUT/name, target)
        public_paths.append(target)
    target = PUBLIC/'fig09-interior-a.html'
    shutil.copyfile(OUT/'fig09-interior-a.html', target)
    public_paths.append(target)
    for name in ['fig09-clean-a.html', 'fig09-frames.html']:
        page = PUBLIC/name
        text = page.read_text()
        if 'id="interior-candidates"' not in text:
            end = text.index('</h1>') + len('</h1>')
            text = text[:end] + '<p id="interior-candidates"><strong>新しい候補：</strong><a href="fig09-interior-a.html">全6コマがホログラム内に収まる、境界で見切れない(a)の6候補を見る</a></p>' + text[end:]
            page.write_text(text)
        public_paths.append(page)
    receipt = json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)): sha(p) for p in public_paths})
    receipt['total_bytes'] = sum((PUBLIC/name).stat().st_size for name in receipt['files'])
    temp = PREVIEW/'public-manifest.interior-a.tmp'
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(PREVIEW/'public-manifest.json')
    assert protected == {p: sha(p) for p in protected}
    write('verification.json', dict(status='PASS', cases=6, panels=36, markers=60,
        all_crops_fully_recorded=True, protected_paper_outputs=protected,
        published_paths=[str(p.relative_to(PUBLIC)) for p in public_paths]))
    print(json.dumps({'status':'PASS', 'cases':6, 'panels':36, 'published_files':len(public_paths), 'zip_entries':78}, indent=2))


if __name__ == '__main__':
    main()
