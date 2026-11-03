"""Author-confirmed examples 03/08, with large per-frame views and a larger paper plate."""
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pypdf import PdfWriter
from common import *

SELECTED=[('a','03','15:6509:6639:6750'),('b','08','22:3758:4041:4087')]
COLORS=['#d94235','#2674b9']
SIDE_IN=2.1
WIDTH_IN=7.25


def image_panel(ax,case,rec,groups,scale=False,standalone=False):
    data=np.load(PRIVATE/'gallery_images'/rec['cache'])
    assert sha(PRIVATE/'gallery_images'/rec['cache'])==rec['sha256']
    ax.imshow(data['display'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    for group,color in zip(groups,COLORS):
        tail=group.loc[group.frame.between(rec['frame']-10,rec['frame'])]
        ax.plot(tail.x-rec['x0'],tail.y-rec['y0'],color=color,lw=.9 if not standalone else 1.25)
        point=group.loc[group.frame.eq(rec['frame'])]
        if len(point):
            p=point.iloc[0];radius=max(2,p.diameter_px/2)
            ax.add_patch(Circle((p.x-rec['x0'],p.y-rec['y0']),radius,fill=False,color=color,lw=1 if not standalone else 1.35))
    if scale:
        x=70 if case['option']=='08' else 8
        ax.plot([x,x+50],[117,117],color='#263238',lw=3.5)
        ax.plot([x,x+50],[117,117],color='white',lw=2)
        ax.text(x+25,112,'500 µm',ha='center',va='bottom',fontsize=7.4 if not standalone else 10,
                color='#263238',bbox=dict(facecolor='white',alpha=.86,edgecolor='none',pad=.35))
    ax.set(xlim=(-.5,127.5),ylim=(127.5,-.5),xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set(linewidth=.65,color='#263238')


def trajectories(ax,case,groups,large=False):
    for group,color in zip(groups,COLORS):
        part=group.loc[group.frame.between(case['display_t0_frame']-30,case['display_t0_frame']+20)]
        xx,zz,yy=part.x.to_numpy()*.01,part.z.to_numpy()*.1,part.y.to_numpy()*.01
        ax.plot(xx,zz,yy,color=color,lw=.8,ls='--',zorder=2)
        ax.scatter(xx,zz,yy,s=7 if not large else 15,color=color,edgecolor='white',linewidth=.25,depthshade=False,zorder=3)
        for i,m in [(0,'^'),(-1,'s')]:ax.scatter([xx[i]],[zz[i]],[yy[i]],marker=m,s=24 if not large else 42,color=color,edgecolor='#263238',linewidth=.5,depthshade=False,zorder=5)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(MaxNLocator(3));axis.pane.fill=False;axis._axinfo['grid']['linewidth']=.35
    ax.set_xlabel('x [mm]',fontsize=7 if not large else 10,labelpad=-4 if not large else 0)
    ax.set_ylabel('z [mm]',fontsize=7 if not large else 10,labelpad=-4 if not large else 0)
    ax.set_zlabel('y [mm]',fontsize=7 if not large else 10,labelpad=-3 if not large else 1)
    ax.tick_params(labelsize=6.2 if not large else 8,pad=-1 if not large else 0)
    ax.invert_zaxis();ax.view_init(elev=21,azim=-60)
    ax.set_title('Three-dimensional trajectories',fontsize=7.4 if not large else 11,pad=2)


def plate(cases,groups_by_id,target):
    h=5.2*len(cases)+.4
    fig=plt.figure(figsize=(WIDTH_IN,h),facecolor='white')
    handles=[Line2D([],[],color=c,marker='o',markerfacecolor='none',lw=1,label=l)
             for c,l in zip(COLORS,['Terminating track (A)','Continuing track (B)'])]
    handles += [Line2D([],[],marker=m,lw=0,color='#263238',label=l) for m,l in [('^','Track start'),('s','Track end')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,
               fontsize=8,handlelength=1.5,columnspacing=1.3,borderaxespad=.25)
    layout=[]
    for row,case in enumerate(cases):
        top=h-.48-row*5.2;groups=groups_by_id[case['candidate_id']]
        d1,d2=case['parent_diameters_um']
        fig.text(.5,top/h,f'({case["case"]}) {d1:.1f} + {d2:.1f} → {case["post_diameter_um"]:.1f} µm',
                 ha='center',va='top',fontsize=11.4)
        upper=top-.37-SIDE_IN;lower=upper-SIDE_IN-.30
        positions=[(.175,upper),(2.575,upper),(4.975,upper),(2.575,lower),(4.975,lower)]
        for k,(rec,(x,y)) in enumerate(zip(case['frames'],positions)):
            ax=fig.add_axes([x/WIDTH_IN,y/h,SIDE_IN/WIDTH_IN,SIDE_IN/h])
            image_panel(ax,case,rec,groups,scale=k==0)
            time=rec['offset_ms'];ax.set_title('0 ms' if time==0 else f'{time:+.0f} ms',fontsize=9,pad=4)
            layout.append(dict(case=case['case'],frame=rec['frame'],width_inches=SIDE_IN,
                               height_inches=SIDE_IN,relative_width=SIDE_IN/WIDTH_IN))
        ax=fig.add_axes([.09/WIDTH_IN,(lower+.10)/h,2.23/WIDTH_IN,1.94/h],projection='3d',computed_zorder=False)
        trajectories(ax,case,groups)
    fig.savefig(target.with_suffix('.pdf'),facecolor='white')
    fig.savefig(target.with_suffix('.png'),dpi=300,facecolor='white')
    plt.close(fig)
    return layout


def main(rebuild_collection=True):
    target=PACKAGE/'fig10-selected';target.mkdir(exist_ok=True)
    originals=read(PRIVATE/'gallery_images/manifest.json')
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet').set_index('candidate_id')
    cases=[];groups_by_id={};source_hashes={}
    for panel,option,cid in SELECTED:
        source=next(c for c in originals['cases'] if c['case']==option)
        assert source['candidate_id']==cid and labels.loc[cid,'status']=='collision'
        case={**source,'case':panel,'option':option};cases.append(case)
        path=CAMPAIGN/f'scenes/scene{case["scene_num"]:04d}/tracking/smoothed.parquet'
        tracks=pd.read_parquet(path);event=labels.loc[cid]
        groups_by_id[cid]=[tracks.loc[tracks.track_id.eq(tid)].sort_values('frame') for tid in [event.vanished_track_id,event.survivor_track_id]]
        source_hashes[str(case['scene_num'])]=sha(path)
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    layout=plate(cases,groups_by_id,FIGURES/'fig10')
    for c in cases:
        plate([c],groups_by_id,target/c['case'])
        for k,rec in enumerate(c['frames'],1):
            fig=plt.figure(figsize=(3,3));ax=fig.add_axes([0,0,1,1])
            image_panel(ax,c,rec,groups_by_id[c['candidate_id']],scale=k==1,standalone=True)
            fig.savefig(target/f'{c["case"]}-frame{k}.png',dpi=256,facecolor='white');plt.close(fig)
        fig=plt.figure(figsize=(4,4));ax=fig.add_axes([.04,.06,.87,.86],projection='3d',computed_zorder=False)
        trajectories(ax,c,groups_by_id[c['candidate_id']],large=True)
        fig.savefig(target/f'{c["case"]}-trajectories.png',dpi=220,facecolor='white');plt.close(fig)
    for source,out,scale in [(FIGURES/'fig10.pdf',PACKAGE/'preview/fig10',1800)]+[(target/f'{c["case"]}.pdf',target/f'{c["case"]}-preview',1800) for c in cases]:
        subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to',str(scale),'-png',str(source),str(out)],check=True)
    geometry=dict(panel_width_relative=SIDE_IN/WIDTH_IN,previous_gallery_panel_width_relative=.175,
                  linear_display_gain_at_equal_figure_width=(SIDE_IN/WIDTH_IN)/.175,
                  crop_pixels=128,pixel_pitch_um=10,web_frame_columns_desktop=3,web_frame_columns_phone=1)
    provenance=dict(created_at=stamp(),status='author_confirmed',selected_options=['03','08'],cases=cases,
        model_sha256=MODEL_SHA,track_sha256=source_hashes,display_geometry=geometry,paper_panels=layout,
        unchanged_source_crops=True,unchanged_measurements=True,
        reference_figure9= {'file':'experiments/amt_figure9_refinement_20261102/results/confirmed-a/fig09-a-confirmed.pdf',
                           'sha256':sha(REPO/'experiments/amt_figure9_refinement_20261102/results/confirmed-a/fig09-a-confirmed.pdf')},
        selection_note='User confirmed options 03 and 08. Five displayed times are retained; 128px crops are enlarged in the layout, without recropping or reconstruction.')
    save(DATA/'figure10_provenance.json',provenance);save(target/'selection.json',provenance)
    write_page(cases,geometry)
    # Keep the earlier option gallery usable, but make the final decision clear.
    page=PACKAGE/'fig10-candidates.html'
    if page.exists():
        text=page.read_text().replace('現在のFig.10の2例は候補04・07です。既存のFig.10は維持し、どの例にも差し替えられる単独PDFとPNGを用意しています。',
            'Fig.10は候補03・08で確定しました。<a href="fig10-selected.html">確定版の大きな各コマ表示</a>を開けます。候補04・07は選択前のFig.10の例です。')
        text=text.replace('現在のFig.10(b)','選択前のFig.10(b)').replace('現在のFig.10(a)','選択前のFig.10(a)')
        page.write_text(text)
    if rebuild_collection:
        writer=PdfWriter()
        for n in [4,5,10,11,12,13,14,15]:writer.append(FIGURES/f'fig{n:02d}.pdf',outline_item=f'Figure {n}')
        writer.write(PACKAGE/'experimental_figures.pdf')
    print(f'Confirmed 03/08; paper-panel width at equal figure width is {geometry["linear_display_gain_at_equal_figure_width"]:.3f} times the candidate layout.')


def write_page(cases,geometry):
    blocks=[]
    for c in cases:
        tag=c['case'];option=c['option'];d1,d2=c['parent_diameters_um']
        frames=[]
        for k,r in enumerate(c['frames'],1):
            time='0 ms' if not r['offset_ms'] else f'{r["offset_ms"]:+.0f} ms'
            frames.append(f'<figure><figcaption><strong>{time}</strong><span>f={r["frame"]}</span></figcaption><a href="fig10-selected/{tag}-frame{k}.png"><img loading="lazy" src="fig10-selected/{tag}-frame{k}.png" alt="確定例{tag}の{time}、軌跡表示付き"></a><p><a href="fig10-selected/{tag}-frame{k}.png">拡大画像</a> · <a href="gallery/{option}-frame{k}.png">元の128画素</a></p></figure>')
        frames.append(f'<figure class="trajectory-view"><figcaption><strong>3次元軌跡</strong></figcaption><a href="fig10-selected/{tag}-trajectories.png"><img loading="lazy" src="fig10-selected/{tag}-trajectories.png" alt="確定例{tag}の3次元軌跡"></a><p>▲ 軌跡の始点　■ 終点</p></figure>')
        blocks.append(f'''<section id="case-{tag}"><div class="eyebrow">CONFIRMED ({tag.upper()}) · OPTION {option}</div><h2>候補{option} · S{c['scene_num']:04d} / {c['review_rank']}位</h2>
<p class="case-diameters">{d1:.1f} + {d2:.1f} → {c['post_diameter_um']:.1f} µm</p>
<p>赤：消失側A、青：存続側B。時刻0は候補フレーム{c['candidate_frame_raw']}です。</p><div class="large-frames">{''.join(frames)}</div>
<div class="buttons"><a class="button" href="fig10-selected/{tag}.pdf">この例のPDF</a><a class="button" href="fig10-selected/{tag}.png">この例の300 dpi PNG</a><a class="button" href="gallery/{option}.gif">レビュー時のGIF</a></div>
<details><summary>GIFをここで見る</summary><img class="review-gif" loading="lazy" src="gallery/{option}.gif" alt="確定例{tag}のレビュー動画"><p>レビューGIFはAが金色、Bがシアンです。</p></details></section>''')
    page=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fig.10 確定版03・08｜拡大ビュー</title><link rel="stylesheet" href="report.css"><link rel="stylesheet" href="fig10-selected.css"></head><body class="selected-page">
<header><div class="eyebrow">FIGURE 10 · AUTHOR CONFIRMED</div><h1>03・08で確定。<br>各コマを大きく表示</h1><p>Fig.9のように各コマを独立した画像として並べ、スマートフォンでは1列で表示します。
画像をタップすると拡大画像を開けます。両親粒子が収まる128×128画素と、−2・−1・0・+1・+2 msの5時刻を維持しました。</p>
<div class="buttons"><a class="button primary" href="figures/fig10.pdf">確定Fig.10 PDF</a><a class="button" href="figures/fig10.png">300 dpi PNG</a><a class="button" href="text/caption_fig10.tex">英文キャプション</a><a class="button" href="index.html#fig10">実験節の報告へ</a></div></header>
<nav><div><a href="#case-a">(a) 候補03</a><a href="#case-b">(b) 候補08</a><a href="#paper">論文用の全体図</a></div></nav><main>{''.join(blocks)}
<section id="paper"><h2>論文用の確定Fig.10</h2><p>各コマの幅を、図全体の幅に対して17.5%から29.0%へ拡大しました。同じ横幅で表示すると約{geometry['linear_display_gain_at_equal_figure_width']:.2f}倍です。
5コマと3次元軌跡を3列・2段に配置しています。</p><figure><a href="figures/fig10.pdf"><img class="paper-plate" loading="lazy" src="preview/fig10.png" alt="確定Fig.10全体"></a></figure>
<p><a href="fig10-selected/selection.json">選択例・フレーム・表示条件の記録</a> · <a href="fig10-candidates.html">候補10例の一覧</a></p></section>
<footer>生成: {stamp()} · 元の再構成像、測定値、レビュー判定は保持しています。Quick Tunnelの停止でリンクは利用できなくなります。</footer></main></body></html>'''
    (PACKAGE/'fig10-selected.html').write_text(page)
    shutil.copy2(Path(__file__).with_name('fig10-selected.css'),PACKAGE/'fig10-selected.css')


if __name__=='__main__':main()
