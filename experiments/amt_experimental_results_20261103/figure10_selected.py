"""Author-confirmed 03/08: original panel layout, enlarged image field only."""
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
WIDTH_IN=12.8
VIEW_PIXELS={'03':96,'08':120}


def view_bounds(case):
    width=VIEW_PIXELS[case['option']]
    return 63.5-width/2,63.5+width/2


def image_panel(ax,case,rec,groups,scale=False,standalone=False):
    data=np.load(PRIVATE/'gallery_images'/rec['cache'])
    assert sha(PRIVATE/'gallery_images'/rec['cache'])==rec['sha256']
    ax.imshow(data['display'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    for group,color in zip(groups,COLORS):
        tail=group.loc[group.frame.between(rec['frame']-10,rec['frame'])]
        ax.plot(tail.x-rec['x0'],tail.y-rec['y0'],color=color,lw=1.1 if not standalone else 1.25)
        point=group.loc[group.frame.eq(rec['frame'])]
        if len(point):
            p=point.iloc[0];radius=max(2,p.diameter_px/2)
            ax.add_patch(Circle((p.x-rec['x0'],p.y-rec['y0']),radius,fill=False,color=color,lw=1.2 if not standalone else 1.35))
    low,high=view_bounds(case)
    if scale:
        x=high-56 if case['option']=='08' else low+3
        y=high-10 if case['option']=='08' else high-4
        ax.plot([x,x+50],[y,y],color='black',lw=3.3)
        ax.plot([x,x+50],[y,y],color='white',lw=2)
        ax.text(x+25,y-5,'500 µm',ha='center',va='bottom',fontsize=9 if not standalone else 10,
                color='#263238',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.4))
    ax.set(xlim=(low,high),ylim=(high,low),xticks=[],yticks=[])


def trajectories(ax,case,groups,large=False):
    for group,color in zip(groups,COLORS):
        part=group.loc[group.frame.between(case['display_t0_frame']-30,case['display_t0_frame']+20)]
        xx,zz,yy=part.x.to_numpy()*.01,part.z.to_numpy()*.1,part.y.to_numpy()*.01
        ax.plot(xx,zz,yy,color=color,lw=1.2,ls='--',zorder=2)
        ax.scatter(xx,zz,yy,s=12,color=color,edgecolor='white',linewidth=.3,depthshade=False,zorder=3)
        for i,m in [(0,'^'),(-1,'s')]:ax.scatter([xx[i]],[zz[i]],[yy[i]],marker=m,s=46,color=color,edgecolor='#263238',linewidth=.7,depthshade=False,zorder=5)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(MaxNLocator(3));axis.pane.fill=False;axis._axinfo['grid']['linewidth']=.4
    ax.set_xlabel('x [mm]',labelpad=5);ax.set_ylabel('z [mm]',labelpad=5);ax.set_zlabel('y [mm]',labelpad=7)
    ax.tick_params(labelsize=9,pad=1)
    ax.invert_zaxis();ax.view_init(elev=21,azim=-60)
    legend=[Line2D([],[],marker='o',lw=0,color=c,label=l) for c,l in zip(COLORS,['Terminating particle','Continuing particle'])]
    legend += [Line2D([],[],marker=m,lw=0,color='#263238',label=l) for m,l in [('^','Start'),('s','End')]]
    ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(.02,1.11),frameon=False,fontsize=9.5,handletextpad=.3,labelspacing=.25)


def plate(cases,groups_by_id,target):
    h=12.1 if len(cases)==2 else 6.05
    fig=plt.figure(figsize=(WIDTH_IN,h),facecolor='white')
    # The combined page exactly reuses the original axes/title coordinates.
    # A single-case export is the corresponding half-page, not a new layout.
    def yy(y):return y if len(cases)==2 else 2*(y-.5)
    def hh(height):return height if len(cases)==2 else height*2
    layout=[]
    for row,case in enumerate(cases):
        top=.96-row*.49;bottom=top-.425;groups=groups_by_id[case['candidate_id']]
        d1,d2=case['parent_diameters_um']
        fig.text(.5,yy(top+.012),f'{d1:.1f} + {d2:.1f} → {case["post_diameter_um"]:.1f} µm',
                 ha='center',va='bottom',fontsize=17)
        ax=fig.add_axes([.018,yy(bottom+.025),.375,hh(.35)],projection='3d',computed_zorder=False)
        trajectories(ax,case,groups)
        fig.text(.205,yy(bottom+.006),'Smoothed three-dimensional trajectories',ha='center',fontsize=11)
        positions=[(.425,top-.191),(.615,top-.191),(.805,top-.191),(.520,top-.398),(.710,top-.398)]
        for k,(rec,(x,y)) in enumerate(zip(case['frames'],positions)):
            bounds=[x,yy(y),.175,hh(.185)]
            ax=fig.add_axes(bounds)
            image_panel(ax,case,rec,groups,scale=k==0)
            time=rec['offset_ms'];ax.set_xlabel('0 ms' if time==0 else f'{time:+.2f} ms',fontsize=11,labelpad=4)
            layout.append(dict(case=case['case'],frame=rec['frame'],axes_bounds=bounds,
                               relative_width=.175,view_bounds_pixels=list(view_bounds(case))))
        fig.text(.5,yy(bottom-.020),f'({case["case"]})',ha='center',fontsize=13)
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
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':11})
    margins=[]
    for c in cases:
        low,high=view_bounds(c)
        for rec in c['frames']:
            for group in groups_by_id[c['candidate_id']]:
                for p in group.loc[group.frame.eq(rec['frame'])].itertuples():
                    x=p.x-rec['x0'];y=p.y-rec['y0'];radius=max(2,p.diameter_px/2)
                    margin=min(x-radius-low,high-x-radius,y-radius-low,high-y-radius)
                    assert margin>0,(c['case'],rec['frame'],p.track_id,margin)
                    margins.append(dict(case=c['case'],frame=rec['frame'],track_id=int(p.track_id),
                        minimum_disk_margin_pixels=float(margin)))
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
    geometry=dict(figure_size_inches=[12.8,12.1],panel_width_relative=.175,
                  layout_reference_commit='2310bfc',layout_restored=True,
                  source_crop_pixels=128,pixel_pitch_um=10,view_pixels=VIEW_PIXELS,
                  in_panel_magnification={k:128/v for k,v in VIEW_PIXELS.items()},
                  fixed_scale_within_each_case=True,observed_disk_margins=margins,
                  limitation='Option 08 at -2 ms spans 117.97 pixels across both observed disks; a 96px view would clip a parent. Its fixed 120px view preserves both.')
    provenance=dict(created_at=stamp(),status='author_confirmed',selected_options=['03','08'],cases=cases,
        model_sha256=MODEL_SHA,track_sha256=source_hashes,display_geometry=geometry,paper_panels=layout,
        unchanged_source_crops=True,unchanged_measurements=True,
        reference_figure9= {'file':'experiments/amt_figure9_refinement_20261102/results/confirmed-a/fig09-a-confirmed.pdf',
                           'sha256':sha(REPO/'experiments/amt_figure9_refinement_20261102/results/confirmed-a/fig09-a-confirmed.pdf')},
        selection_note='User confirmed options 03 and 08 and requested larger particles within existing panels. Original layout and five times retained; only view limits narrowed to 96px/120px using unchanged 128px cached reconstructions.')
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
    print('Confirmed 03/08; original layout restored. Image views: 96px (1.333x) / 120px (1.067x).')


def write_page(cases,geometry):
    blocks=[]
    for c in cases:
        tag=c['case'];option=c['option'];d1,d2=c['parent_diameters_um']
        width=VIEW_PIXELS[option];gain=128/width
        limitation='最初の−2 msで親粒子が上下に離れるため、この例は1.07倍までの拡大です。5コマすべて同じ物理スケールにそろえています。' if option=='08' else '5コマすべて同じ物理スケールにそろえています。'
        blocks.append(f'''<section id="case-{tag}"><div class="eyebrow">CONFIRMED ({tag.upper()}) · OPTION {option}</div><h2>候補{option} · S{c['scene_num']:04d} / {c['review_rank']}位</h2>
<p class="case-diameters">{d1:.1f} + {d2:.1f} → {c['post_diameter_um']:.1f} µm</p>
<p>画像枠内の表示範囲を128画素から{width}画素（{width*.01:.2f} mm）に狭め、粒子を{gain:.2f}倍に拡大しました。{limitation}</p>
<figure><a href="fig10-selected/{tag}.png"><img class="original-layout" loading="lazy" src="fig10-selected/{tag}-preview.png" alt="候補{option}：元の配置を保ち、画像枠内の表示範囲のみ拡大"></a></figure>
<div class="buttons"><a class="button" href="fig10-selected/{tag}.pdf">この例のPDF</a><a class="button" href="fig10-selected/{tag}.png">この例の300 dpi PNG</a><a class="button" href="gallery/{option}.gif">レビュー時のGIF</a></div>
<details><summary>拡大前の128画素の視野</summary><img class="original-layout" loading="lazy" src="gallery/{option}-preview.png" alt="候補{option}の拡大前の視野"><p>元の画像と計測値を保持しています。現在の図は同じ画像の表示範囲だけを変更しています。</p></details>
<details><summary>GIFをここで見る</summary><img class="review-gif" loading="lazy" src="gallery/{option}.gif" alt="確定例{tag}のレビュー動画"><p>レビューGIFはAが金色、Bがシアンです。</p></details></section>''')
    page=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fig.10 確定版03・08｜拡大ビュー</title><link rel="stylesheet" href="report.css"><link rel="stylesheet" href="fig10-selected.css"></head><body class="selected-page">
<header><div class="eyebrow">FIGURE 10 · AUTHOR CONFIRMED</div><h1>03・08で確定。<br>元の配置で粒子を拡大</h1><p>左に3次元軌跡、右に5コマという元の配置を戻しました。
画像枠の大きさと位置は維持し、枠内の視野だけを拡大しています。−2・−1・0・+1・+2 msの5時刻は同じです。</p>
<div class="buttons"><a class="button primary" href="figures/fig10.pdf">確定Fig.10 PDF</a><a class="button" href="figures/fig10.png">300 dpi PNG</a><a class="button" href="text/caption_fig10.tex">英文キャプション</a><a class="button" href="index.html#fig10">実験節の報告へ</a></div></header>
<nav><div><a href="#case-a">(a) 候補03</a><a href="#case-b">(b) 候補08</a><a href="#paper">論文用の全体図</a></div></nav><main>{''.join(blocks)}
<section id="paper"><h2>論文用の確定Fig.10</h2><p>図の寸法、画像枠の位置・大きさ、3次元軌跡の配置を元に戻しました。03は1.33倍、08は1.07倍の視野拡大です。
500 µmのスケールバーは表示範囲に合わせて正しい長さで描いています。全10コマの観測粒子円が画像枠内に収まることを確認しました。</p><figure><a href="figures/fig10.pdf"><img class="paper-plate" loading="lazy" src="preview/fig10.png" alt="確定Fig.10全体"></a></figure>
<p><a href="fig10-selected/selection.json">選択例・フレーム・表示条件の記録</a> · <a href="fig10-candidates.html">候補10例の一覧</a></p></section>
<footer>生成: {stamp()} · 元の再構成像、測定値、レビュー判定は保持しています。Quick Tunnelの停止でリンクは利用できなくなります。</footer></main></body></html>'''
    (PACKAGE/'fig10-selected.html').write_text(page)
    shutil.copy2(Path(__file__).with_name('fig10-selected.css'),PACKAGE/'fig10-selected.css')


if __name__=='__main__':main()
