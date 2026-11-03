"""Ten Fig.10 illustration options; human labels and analysis anchors stay fixed."""
import html
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfWriter
from common import *

SELECTION=[
    ('12:5450:4750:6588','親粒径がほぼ等しい例。接近から合体後までを比較しやすい。'),
    ('13:10153:10312:5913','同程度の粒径の例。合体直前の2粒子と合体後の形状を追える。'),
    ('15:6509:6639:6750','粒径差のある例。小さい粒子が大きい粒子に近づく過程が長く見える。'),
    ('17:10138:9881:8565','選択前のFig.10(b)と同じ物理例。大きい方の親が消失側の軌跡に対応する。'),
    ('18:5198:4981:4474','親粒径が近く、合体直前から直後に形状が変わる例。'),
    ('20:3702:4099:5429','接近と合体後の単一粒子を同じ領域で比較しやすい例。'),
    ('21:2906:3007:3612','選択前のFig.10(a)と同じ物理例。接近と合体直後の変形を比較できる。'),
    ('22:3758:4041:4087','粒径差が大きい例。小さい親は約100 µm、大きい親は約211 µm。'),
    ('25:6128:6159:6903','今回の候補集では大きい粒子同士の例。候補時刻後も変形が続く。'),
    ('28:6577:6818:6810','粒径差のある例。小さい親の接近から合体後までを追える。'),
]


def panel(case,event,tracks,target):
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(12.8,7.0),facecolor='white')
    colors=['#d94235','#2674b9']
    groups=[tracks.loc[tracks.track_id.eq(tid)].sort_values('frame') for tid in [event.vanished_track_id,event.survivor_track_id]]
    d1,d2=case['parent_diameters_um']
    fig.text(.5,.967,f'Candidate {case["case"]}  ·  S{case["scene_num"]:04d} / rank {case["review_rank"]}',ha='center',va='top',fontsize=15)
    fig.text(.5,.917,f'{d1:.1f} + {d2:.1f} → {case["post_diameter_um"]:.1f} µm',ha='center',va='top',fontsize=17)
    ax=fig.add_axes([.018,.21,.37,.53],projection='3d',computed_zorder=False)
    for group,color in zip(groups,colors):
        part=group.loc[group.frame.between(case['display_t0_frame']-30,case['display_t0_frame']+20)]
        xx,zz,yy=part.x.to_numpy()*.01,part.z.to_numpy()*.1,part.y.to_numpy()*.01
        ax.plot(xx,zz,yy,color=color,lw=1.2,ls='--',zorder=2)
        ax.scatter(xx,zz,yy,s=13,color=color,edgecolor='white',linewidth=.3,depthshade=False,zorder=3)
        for k,m in [(0,'^'),(-1,'s')]:ax.scatter([xx[k]],[zz[k]],[yy[k]],s=48,marker=m,color=color,edgecolor='#263238',linewidth=.7,depthshade=False,zorder=5)
    ax.set_xlabel('x [mm]',labelpad=5);ax.set_ylabel('z [mm]',labelpad=5);ax.set_zlabel('y [mm]',labelpad=7)
    ax.invert_zaxis();ax.view_init(elev=21,azim=-60)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(MaxNLocator(3));axis.pane.fill=False;axis._axinfo['grid']['linewidth']=.4
    ax.tick_params(labelsize=9,pad=1)
    legend=[Line2D([],[],marker='o',lw=0,color=c,label=l) for c,l in zip(colors,['Terminating track (A)','Continuing track (B)'])]
    legend += [Line2D([],[],marker=m,lw=0,color='#263238',label=l) for m,l in [('^','Start'),('s','End')]]
    ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(.02,1.08),frameon=False,fontsize=9.5,labelspacing=.3)
    fig.text(.205,.155,'Smoothed three-dimensional trajectories',ha='center',fontsize=10.5)
    positions=[(.425,.53),(.615,.53),(.805,.53),(.520,.16),(.710,.16)]
    for k,(rec,(left,low)) in enumerate(zip(case['frames'],positions)):
        data=np.load(PRIVATE/'gallery_images'/rec['cache'])
        image_ax=fig.add_axes([left,low,.175,.32])
        image_ax.imshow(data['display'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        frame=rec['frame']
        for group,color in zip(groups,colors):
            tail=group.loc[group.frame.between(frame-10,frame)]
            image_ax.plot(tail.x-rec['x0'],tail.y-rec['y0'],color=color,lw=1.1)
            point=group.loc[group.frame.eq(frame)]
            if len(point):
                p=point.iloc[0]
                image_ax.add_patch(Circle((p.x-rec['x0'],p.y-rec['y0']),max(2,p.diameter_px/2),fill=False,edgecolor=color,lw=1.2))
        if k==0:
            # The first view of option 08 places the small parent beside the
            # usual lower-left scale bar; keep the annotation clear of it.
            bar_start=70 if case['case']=='08' else 8
            image_ax.plot([bar_start,bar_start+50],[117,117],color='black',lw=3.3)
            image_ax.plot([bar_start,bar_start+50],[117,117],color='white',lw=2)
            image_ax.text(bar_start+25,112,'500 µm',ha='center',va='bottom',fontsize=9,color='#263238',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.4))
        image_ax.set_xlim(-.5,127.5);image_ax.set_ylim(127.5,-.5);image_ax.set_xticks([]);image_ax.set_yticks([])
        image_ax.set_xlabel(f'{rec["offset_ms"]:+.0f} ms' if rec['offset_ms'] else '0 ms',labelpad=5,fontsize=11)
        Image.fromarray(data['display']).save(target/f'{case["case"]}-frame{k+1}.png')
    fig.text(.5,.055,f'ID {case["candidate_id"]}  ·  t = 0: raw candidate frame {case["candidate_frame_raw"]} (4000 fps)',ha='center',fontsize=10.5)
    fig.text(.5,.027,'Phase retrieval: 3 iterations. Relative axial depth. Each crop: 128 × 128 px; 10 µm / px.',ha='center',fontsize=9.5,color='#465561')
    for ext in ['pdf','png']:fig.savefig(target/f'{case["case"]}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    with Image.open(target/f'{case["case"]}.png') as im:
        im.thumbnail((1600,1600));im.save(target/f'{case["case"]}-preview.png')


def main():
    from build_report import table
    manifest=read(PRIVATE/'gallery_images/manifest.json')
    assert [c['candidate_id'] for c in manifest['cases']]==[c[0] for c in SELECTION]
    target=PACKAGE/'gallery';target.mkdir(exist_ok=True)
    events=pd.read_parquet(PRIVATE/'physical_events_all.parquet').set_index('candidate_id',drop=False)
    rows=[];cards=[];book=PdfWriter()
    for case,(_,note) in zip(manifest['cases'],SELECTION):
        event=events.loc[case['candidate_id']]
        assert event.status=='collision' and event.model_sha256==MODEL_SHA
        root=CAMPAIGN/f'scenes/scene{int(event.scene_num):04d}'
        review=next(r for r in read(root/'review/manifest.json')['candidates'] if r['candidate_id']==case['candidate_id'])
        assert sha(CAMPAIGN/review['asset'])==event.asset_sha256==case['review_asset_sha256']
        for frame in case['frames']:assert sha(PRIVATE/'gallery_images'/frame['cache'])==frame['sha256']
        panel(case,event,pd.read_parquet(root/'tracking/smoothed.parquet'),target)
        tag=case['case'];shutil.copy2(CAMPAIGN/review['asset'],target/f'{tag}.gif')
        book.append(target/f'{tag}.pdf',outline_item=f'Candidate {tag}: S{int(event.scene_num):04d} / rank {int(event.review_rank)}')
        rows.append(dict(option=tag,candidate_id=event.candidate_id,scene_num=int(event.scene_num),review_rank=int(event.review_rank),
            candidate_frame_raw=int(event.candidate_frame_raw),score=float(event.score),
            parent_A_um=float(event.pre_diameter_vanished_um),parent_B_um=float(event.pre_diameter_survivor_um),post_um=float(event.post_diameter_um),
            expected_post_diameter_um=float(event.expected_post_diameter_um),diameter_available=bool(event.diameter_available),velocity_available=bool(event.velocity_available),
            model_sha256=MODEL_SHA,review_asset_sha256=event.asset_sha256,selection_note_ja=note))
        frames=''.join(f'<a href="gallery/{tag}-frame{k+1}.png"><img loading="lazy" src="gallery/{tag}-frame{k+1}.png" alt="候補{tag}、{r["offset_ms"]:+.0f} msの再構成像"><span>{r["offset_ms"]:+.0f} ms / f={r["frame"]}</span></a>' for k,r in enumerate(case['frames']))
        caption=(f'Visually confirmed collision–coalescence candidate {tag} from S{int(event.scene_num):04d}, review rank {int(event.review_rank)}. '
                 f'Terminating-parent, continuing-parent and post-event median diameters are {event.pre_diameter_vanished_um:.1f}, {event.pre_diameter_survivor_um:.1f} and {event.post_diameter_um:.1f} µm. '
                 f'Time zero is raw candidate frame {int(event.candidate_frame_raw)}; image offsets are −8, −4, 0, +4 and +8 frames at 4000 fps. '
                 'Red and blue denote terminating and continuing tracks, respectively. Triangles/squares mark starts/ends of displayed trajectory segments. '
                 'Images use three phase-retrieval iterations; the scale bar is 500 µm. The axial coordinate is relative depth from the reconstruction front.')
        (target/f'{tag}-caption.txt').write_text(caption+'\n')
        cards.append(f'''<article class="card" id="candidate-{tag}"><div class="figure-title"><span>CANDIDATE {tag}</span><h2>S{int(event.scene_num):04d} · {int(event.review_rank)}位</h2></div>
<p>{html.escape(note)}</p><p class="numbers">A {event.pre_diameter_vanished_um:.1f} + B {event.pre_diameter_survivor_um:.1f} → 合体後 {event.post_diameter_um:.1f} µm</p>
<figure><a href="gallery/{tag}.pdf"><img loading="lazy" src="gallery/{tag}-preview.png" alt="候補{tag}の軌跡と5時刻の再構成像"></a></figure>
<div class="buttons"><a class="button" href="gallery/{tag}.pdf">図用PDF</a><a class="button" href="gallery/{tag}.png">300 dpi PNG</a><a class="button" href="gallery/{tag}.gif">GIFを開く</a><a class="button" href="gallery/{tag}-caption.txt">英文説明</a></div>
<details><summary>GIFと、書き込みのない各コマを見る</summary><p>GIFはレビュー時の画像です。金色がA、シアンがBを示します。図用PDFではAを赤、Bを青で表示しています。</p><img class="review-gif" loading="lazy" src="gallery/{tag}.gif" alt="候補{tag}のレビュー済み動画"><div class="frame-strip">{frames}</div><p class="muted">各コマは128×128画素、10 µm/画素。クリックして元の切り出し画像を保存できます。</p></details>
<p class="muted">候補ID <code>{event.candidate_id}</code> · t=0 は候補時刻f={int(event.candidate_frame_raw)}。
レビュー判定は collision。最終モデルの順位・親粒径・表示画像の対応を保存しています。</p></article>''')
    book.write(PACKAGE/'fig10-candidates.pdf')
    pd.DataFrame(rows).to_csv(DATA/'figure10_candidates.csv',index=False)
    save(DATA/'figure10_candidates_provenance.json',dict(created_at=stamp(),model_sha256=MODEL_SHA,cases=manifest['cases'],
        engine_manifest_sha256=manifest['engine_manifest_sha256'],
        selection='Illustrative purposive selection from human-confirmed cases, checked as image sequences; includes the two existing Figure 10 examples. Not a statistical or residual-based sample.',
        gif_color_convention='A terminating: gold; B continuing: cyan',figure_color_convention='A terminating: red; B continuing: blue'))
    overview=table(['候補','原系列 / 順位','A [µm]','B [µm]','合体後 [µm]'],[[r['option'],f'S{r["scene_num"]:04d} / {r["review_rank"]}位',f'{r["parent_A_um"]:.1f}',f'{r["parent_B_um"]:.1f}',f'{r["post_um"]:.1f}'] for r in rows])
    nav=''.join(f'<a href="#candidate-{i:02d}">{i:02d}</a>' for i in range(1,11))
    page=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fig.10 衝突例の候補10件</title><link rel="stylesheet" href="report.css"></head><body>
<header><div class="eyebrow">FIGURE 10 · TEN ILLUSTRATION OPTIONS</div><h1>Fig.10に使う衝突例の候補10件</h1><p>確認衝突186件から、接近・合体後を画像で比較できる10例を選びました。
Fig.10は候補03・08で確定しました。<a href="fig10-selected.html">確定版の大きな各コマ表示</a>を開けます。候補04・07は選択前のFig.10の例です。</p>
<div class="buttons"><a class="button primary" href="fig10-candidates.pdf">10例をまとめたPDF</a><a class="button" href="index.html#fig10">図表報告へ戻る</a><a class="button" href="data/figure10_candidates.csv">候補一覧CSV</a></div></header>
<nav><div>{nav}</div></nav><main><section><h2>候補の一覧</h2>{overview}
<p>この候補集は図の説明例を選ぶためのものです。母集団を無作為に抽出した標本ではありません。
全例が人のレビューで確認された衝突で、粒径・速度の診断にも使われています。物理残差の小ささによる選別はしていません。</p>
<p>図では時刻0を生の候補時刻に統一し、−2、−1、0、+1、+2 msを示します。
接触が始まる時刻と候補時刻は必ずしも一致しません。元のFig.10は見せ方のための接触時刻を原点としているので、同じ物理例でも表示するコマが異なります。</p>
<p>親粒径の順はA（消失側）＋B（存続側）です。
全例を固定済みコード・位相回復3回で再構成し、同じモデル・候補ID・順位に対応するGIFと照合しました。</p></section>{''.join(cards)}
<footer>生成: {stamp()} · <a href="data/figure10_candidates_provenance.json">表示条件と出典の記録</a> · 人の判定や解析窓は変更していません。</footer></main></body></html>'''
    (PACKAGE/'fig10-candidates.html').write_text(page)
    print('Created 10 candidate PDFs, PNGs, GIFs, 50 unannotated crops and a ten-page PDF.')


if __name__=='__main__':main()
