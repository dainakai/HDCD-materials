#!/usr/bin/env python3
"""Show all six requested frames without score plots or trajectory overlays."""
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

from search import REPO, ROOT, original
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d._core.track_images import point_at

OUT = REPO.parent/'audit/revision-figure9-frames-20261102'
FINAL = REPO.parent/'audit/revision-figure9-refinement-20261102'
PREVIEW = REPO.parent/'audit/revision-review-20261101'
PUBLIC = PREVIEW/'public'
DELTAS = range(-4,2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_label(delta):
    return f'{delta/4:+.2f} ms' if delta else '0.00 ms'


def contact_sheet(label, panels, target):
    # Exact 2x nearest-neighbor display; labels are outside image pixels.
    size, gap, margin, top = 256, 16, 16, 92
    sheet = Image.new('RGB', (2*margin+6*size+5*gap, top+size+38), 'white')
    draw = ImageDraw.Draw(sheet)
    font = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    title_font = ImageFont.truetype(font, 28)
    label_font = ImageFont.truetype(font, 23)
    small_font = ImageFont.truetype(font, 18)
    draw.text((margin,12),label,fill='#17242d',font=title_font)
    for i,p in enumerate(panels):
        x = margin+i*(size+gap)
        draw.text((x,57),f"{p['label']}   {time_label(p['delta'])}",fill='#17242d',font=label_font)
        with Image.open(OUT/'images'/p['filename']) as im:
            enlarged = im.resize((size,size),Image.Resampling.NEAREST).convert('RGB')
            sheet.paste(enlarged,(x,top))
            assert np.array_equal(np.asarray(sheet)[top:top+size,x:x+size][::2,::2,0],np.asarray(im))
        draw.text((x,top+size+9),f"frame {p['frame']}",fill='#53616b',font=small_font)
    sheet.save(target)


def page(cases):
    sections=[]
    for c in cases:
        cards=[]
        for p in c['panels']:
            src='fig09-frames/images/'+p['filename']
            cards.append(f'''<figure><figcaption><strong>{p['label']}</strong><span>{time_label(p['delta'])}</span></figcaption>
<a href="{src}" aria-label="{p['label']} の画像を開く"><img src="{src}" width="128" height="128" alt="{p['label']}、{time_label(p['delta'])} の再構成画像。軌跡の重ね描きなし"></a>
<p>frame {p['frame']} <a href="{src}" download>PNG</a></p></figure>''')
        sections.append(f'''<section id="{c['letter']}"><h2>({c['letter']}) {html.escape(c['title'])}</h2>
<div class="frames">{''.join(cards)}</div><p><a href="fig09-frames/{c['letter']}-all-frames.png">6枚を横一列にした画像を開く</a></p></section>''')
    return '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Fig.9 · 表示フレームの選択</title>
<style>body{margin:0;background:#f6f7f8;color:#17242d;font:16px/1.6 system-ui,sans-serif}main{max-width:1680px;margin:auto;padding:24px}h1{font-size:clamp(24px,4vw,36px);margin:12px 0}h2{font-size:23px;margin:0 0 16px}a{color:#155d8a;text-underline-offset:3px}section{background:#fff;border:1px solid #dce2e6;border-radius:12px;padding:20px;margin:28px 0}.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figcaption{display:flex;justify-content:space-between;gap:4px;padding:0 0 8px;font-variant-numeric:tabular-nums}figure img{width:100%;height:auto;display:block;image-rendering:pixelated;background:#000}figure p{font-size:13px;margin:6px 0;color:#53616b;display:flex;justify-content:space-between}.note{color:#52616b;max-width:80ch}.downloads{display:flex;flex-wrap:wrap;gap:20px}footer{font-size:13px;color:#53616b;margin:32px 0}@media(max-width:1000px){.frames{grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}@media(max-width:560px){main{padding:14px}section{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 12px}figcaption{font-size:15px}}</style></head><body><main>
<a href="synthetic.html#fig9">← Fig.9と結果のページ</a><h1>Fig.9の表示フレームを選ぶ</h1>
<p>−1.00、−0.75、−0.50、−0.25、0.00、＋0.25 msの全6枚です。<strong>A1〜A6、B1〜B6</strong>の番号で指定できます。</p>
<p class="note">画像内に軌跡・マーカー・スコアは重ねていません。現在のFig.9と同じ128×128画素、対象中心のクロップ、継続軌跡の推定深さ、記録画素の1–99パーセンタイルのコントラスト調整です。撮影範囲外は黒表示です。左から時刻順に並び、スマホでは折り返して表示します。</p>
<div class="downloads"><a href="fig09-frames/images.zip" download>12枚のPNGと横並び画像を保存</a><a href="#a">(a) 真の衝突</a><a href="#b">(b) 共有粒子アーティファクト</a></div>
'''+''.join(sections)+'''
<p class="note">このページは表示画像を選ぶための比較用です。論文用Fig.9の採用フレームは、このページの作成では変更していません。PNGは128画素の元サイズ、横並び画像は整数倍の最近傍表示です。</p>
<footer>既存のQuick Tunnelを使用しています。停止すると閲覧できなくなり、再作成するとURLが変わります。</footer></main></body></html>'''


def main():
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    (OUT/'cache').mkdir(exist_ok=True)
    protected=[FINAL/'fig09.pdf',FINAL/'selection.json',REPO.parent/'audit/revision-paper-replacements-20261102/paper-replacements.zip']
    before={str(p):sha(p) for p in protected}
    selected=json.loads((FINAL/'selection.json').read_text())
    candidate_dir=Path(selected['candidate_search_directory'])
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    acquisition=Acquisition.load(ROOT/'config/seed260745-collision.yaml')
    assert acquisition.phase_iterations==3
    torch.set_num_threads(4)
    recon=Reconstructor(acquisition,'cpu')
    cases=[];new_count=0;reused_count=0
    for letter,title,c in zip(['a','b'],['真の衝突','共有粒子アーティファクト'],selected['cases']):
        t=c['candidate_frame']
        tr=tracks.loc[tracks.track_id.eq(c['survivor_track_id'])].sort_values('frame')
        cached=np.load(candidate_dir/f"cache/{c['review_id']}.npz")
        cache_meta=json.loads((candidate_dir/f"cache/{c['review_id']}.json").read_text())
        panels=[]
        for i,d in enumerate(DELTAS,1):
            f=t+d;point=point_at(tr,f,nearest=False)
            assert point is not None,(c['review_id'],f)
            if f'raw_{d}' in cached:
                raw,valid=cached[f'raw_{d}'],cached[f'valid_{d}']
                display=cached[f'display_{d}']
                old=next(p for p in cache_meta['panels'] if p['frame']==f)
                assert np.allclose([old['center_x'],old['center_y'],old['center_z']],[point.x,point.y,point.z])
                x0,y0=old['x0'],old['y0'];reused_count+=1
            else:
                with torch.no_grad():plane=recon.intensity(f,point.z)
                raw,x0,y0,valid=original.crop_with_black_padding(plane,point.x,point.y,128)
                display=original.normalize_valid_crop(raw,valid);new_count+=1
            assert raw.shape==valid.shape==display.shape==(128,128)
            assert np.array_equal(original.normalize_valid_crop(raw,valid),display)
            assert not display[~valid].any()
            assert x0==round(point.x)-64 and y0==round(point.y)-64
            label=f'{letter.upper()}{i}';filename=f'{label}-frame-{f}.png'
            Image.fromarray(display).save(OUT/'images'/filename)
            with Image.open(OUT/'images'/filename) as decoded:
                assert np.array_equal(np.asarray(decoded),display)
            np.savez_compressed(OUT/'cache'/f'{label}.npz',raw=raw,valid=valid,display=display)
            panels.append(dict(label=label,filename=filename,frame=f,delta=d,time_ms=d/4,
                center_x=point.x,center_y=point.y,center_z=point.z,x0=x0,y0=y0,
                recorded_fraction=float(valid.mean()),contrast_low_high=np.percentile(raw[valid],[1,99]).tolist(),
                png_sha256=sha(OUT/'images'/filename),reused_current_fig9_image=f'raw_{d}' in cached))
            print(f'{label}: {time_label(d)}, frame {f}',flush=True)
        case=dict(letter=letter,title=title,review_id=c['review_id'],candidate_frame=t,panels=panels)
        cases.append(case)
        contact_sheet(f"({letter}) "+('True collision' if letter=='a' else 'Shared-particle artifact'),panels,OUT/f'{letter}-all-frames.png')
    assert before=={str(p):sha(p) for p in protected}
    manifest=dict(status='PASS',cases=cases,phase_iterations=3,crop_size_px=128,
        source_selection_sha256=sha(FINAL/'selection.json'),reconstructed_frames=new_count,
        reused_current_fig9_images=reused_count,overlays=False,protected_outputs_unchanged=before)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with zipfile.ZipFile(OUT/'images.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((OUT/'images').glob('*.png')):z.write(p,'images/'+p.name)
        for letter in ['a','b']:z.write(OUT/f'{letter}-all-frames.png',f'{letter}-all-frames.png')
    (OUT/'fig09-frames.html').write_text(page(cases))
    public_paths=[]
    for p in sorted((OUT/'images').glob('*.png')):
        target=PUBLIC/'fig09-frames/images'/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);public_paths.append(target)
    for name in ['a-all-frames.png','b-all-frames.png','images.zip']:
        target=PUBLIC/'fig09-frames'/name;shutil.copyfile(OUT/name,target);public_paths.append(target)
    target=PUBLIC/'fig09-frames.html';shutil.copyfile(OUT/'fig09-frames.html',target);public_paths.append(target)
    receipt=json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)):sha(p) for p in public_paths})
    receipt['total_bytes']=sum((PUBLIC/name).stat().st_size for name in receipt['files'])
    temp=PREVIEW/'public-manifest.frames.tmp'
    temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    print(json.dumps({'status':'PASS','images':12,'new_reconstructions':new_count,'reused_images':reused_count,
        'published_files':len(public_paths),'private_output':str(OUT)},indent=2))


if __name__=='__main__':main()
