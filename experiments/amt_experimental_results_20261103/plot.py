"""Publication figures: historical scientific plot functions with new, audited inputs."""
import sys
import shutil
import importlib.util
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from pypdf import PdfWriter
from common import *
sys.path.insert(0,str(REPO/'scripts'))
sys.path.append('/home/dai-server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from figure_support import population_size as size, population_count as count, velocity, beard1976
from figure_support import review_scores, review_budget, physical_plot, gravity

def savefig(fig,stem,dpi=300):
    fig.savefig(FIGURES/f'{stem}.pdf',facecolor='white')
    fig.savefig(FIGURES/f'{stem}.png',dpi=dpi,facecolor='white')
    plt.close(fig)

def population_velocity_figure():
    """Preserve the manuscript axes, with one actual count normalization for all panels."""
    tracks=pd.read_parquet(PRIVATE/'velocity_tracks.parquet')
    plt.rcdefaults();plt.rcParams.update({'font.size':12,'axes.labelsize':13,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(11.8,5.25));grid=fig.add_gridspec(1,4,width_ratios=[1,1,1,.055],left=.075,right=.92,bottom=.24,top=.84,wspace=.45)
    bins=[]
    for i,(col,name,limits) in enumerate([('vx_m_s','Horizontal velocity',(-.5,.5)),('vy_m_s','Vertical velocity',(-.25,1.)),('vz_m_s','Axial velocity',(-.5,.5))]):
        ax=fig.add_subplot(grid[0,i])
        hb=ax.hexbin(tracks.diameter_um,tracks[col],gridsize=(72,62),extent=(25,300,*limits),mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
        bins.append(hb);ax.set_xlim(25,300);ax.set_ylim(*limits)
        ax.set_xlabel('Particle diameter [µm]');ax.set_ylabel(name+r' [m s$^{-1}$]')
        suffix=' (downward positive)' if i==1 else ''
        ax.text(.5,-.31,f'({chr(97+i)}) {name}'+suffix,transform=ax.transAxes,ha='center',va='top',fontsize=11)
        if i==1:
            diameter=np.linspace(25,300,600)
            terminal=beard1976.terminal_velocity_beard_1976_um_s(diameter,pressure_mb=900.,kelvin=307.182512)/1e6
            curve,=ax.plot(diameter,terminal-.26,'k--',lw=2,label='Beard (1976) $-$ 0.26 m s$^{-1}$')
    norm=LogNorm(vmin=1,vmax=max(float(h.get_array().max()) for h in bins))
    for hb in bins:hb.set_norm(norm)
    cax=fig.add_subplot(grid[0,3]);fig.colorbar(bins[-1],cax=cax).set_label('Tracks per bin')
    fig.legend(handles=[curve],loc='upper center',bbox_to_anchor=(.5,.98),framealpha=.92)
    savefig(fig,'fig05')

def case_figure():
    manifest=read(PRIVATE/'case_images/manifest.json')
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet')
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(12.8,12.1),facecolor='white')
    colors=['#d94235','#2674b9']
    for row,case in enumerate(manifest['cases']):
        top=.96-row*.49
        bottom=top-.425
        event=labels.loc[labels.candidate_id.eq(case['candidate_id'])].iloc[0]
        tracks=pd.read_parquet(CAMPAIGN/f'scenes/scene{int(event.scene_num):04d}/tracking/smoothed.parquet')
        groups=[tracks.loc[tracks.track_id.eq(tid)].sort_values('frame') for tid in [event.vanished_track_id,event.survivor_track_id]]
        d1,d2=case['parent_diameters_um']
        fig.text(.5,top+.012,f'{d1:.1f} + {d2:.1f} → {case["post_diameter_um"]:.1f} µm',ha='center',va='bottom',fontsize=17)
        ax=fig.add_axes([.018,bottom+.025,.375,.35],projection='3d',computed_zorder=False)
        values=[]
        for group,color in zip(groups,colors):
            segment=group.loc[group.frame.between(case['display_t0_frame']-30,case['display_t0_frame']+20)]
            xx=segment.x.to_numpy()*.01;zz=segment.z.to_numpy()*.1;yy=segment.y.to_numpy()*.01
            ax.plot(xx,zz,yy,color=color,lw=1.2,ls='--',zorder=2)
            ax.scatter(xx,zz,yy,s=12,color=color,edgecolor='white',linewidth=.3,depthshade=False,zorder=3)
            for k,marker in [(0,'^'),(-1,'s')]:
                ax.scatter([xx[k]],[zz[k]],[yy[k]],s=46,marker=marker,color=color,edgecolor='#263238',linewidth=.7,depthshade=False,zorder=5)
            values.append(np.c_[xx,zz,yy])
        ax.set_xlabel('x [mm]',labelpad=5);ax.set_ylabel('z [mm]',labelpad=5);ax.set_zlabel('y [mm]',labelpad=7)
        ax.invert_zaxis();ax.view_init(elev=21,azim=-60)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_major_locator(MaxNLocator(3));axis.pane.fill=False;axis._axinfo['grid']['linewidth']=.4
        ax.tick_params(labelsize=9,pad=1)
        legend=[Line2D([],[],marker='o',lw=0,color=c,label=l) for c,l in zip(colors,['Terminating particle','Continuing particle'])]
        legend += [Line2D([],[],marker=m,lw=0,color='#263238',label=l) for m,l in [('^','Start'),('s','End')]]
        ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(.02,1.11),frameon=False,fontsize=9.5,handletextpad=.3,labelspacing=.25)
        fig.text(.205,bottom+.006,'Smoothed three-dimensional trajectories',ha='center',fontsize=11)
        positions=[(.425,top-.191),(.615,top-.191),(.805,top-.191),(.520,top-.398),(.710,top-.398)]
        for k,(rec,(left,low)) in enumerate(zip(case['frames'],positions)):
            data=np.load(PRIVATE/'case_images'/rec['cache'])
            image_ax=fig.add_axes([left,low,.175,.185])
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
                image_ax.plot([8,58],[117,117],color='black',lw=3.3)
                image_ax.plot([8,58],[117,117],color='white',lw=2)
                image_ax.text(33,112,'500 µm',ha='center',va='bottom',fontsize=9,color='#263238',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.4))
            image_ax.set_xlim(-.5,127.5);image_ax.set_ylim(127.5,-.5)
            image_ax.set_xticks([]);image_ax.set_yticks([])
            time=rec['offset_ms'];label='0 ms' if time==0 else f'{time:+.2f} ms'
            image_ax.set_xlabel(label,labelpad=4,fontsize=11)
        fig.text(.5,bottom-.020,f'({case["case"]})',ha='center',fontsize=13)
    savefig(fig,'fig10')
    save(DATA/'figure10_provenance.json',{'cases':[{k:v for k,v in c.items() if k!='frames'} for c in manifest['cases']],
        'frame_lists':[[f['frame'] for f in c['frames']] for c in manifest['cases']],
        'coordinate_note':'z is relative depth from reconstruction front; x/y pixel coordinates times 10 um; y positive downward.',
        'selection':'Same visual experimental events as the prior Fig.10, identified from sequence, image context, position and time; source PDF event filenames are not used as new candidate IDs.'})

def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    panels=ROOT/'panels';panels.mkdir(exist_ok=True)
    plt.rcParams['pdf.fonttype']=42
    bins=pd.read_csv(DATA/'diameter_mean_ci.csv')
    size.plot_distribution(panels/'size',bins.diameter_center_um.to_numpy(),bins['mean'].to_numpy(),bins.ci025.to_numpy(),bins.ci975.to_numpy(),plot_min=25,plot_max=300)
    plt.rcdefaults();count.OUTPUT_DIR=panels;count.style_publication()
    count.render_scene_summary(pl.read_csv(DATA/'scene_population_statistics.csv'))
    spec=importlib.util.spec_from_file_location('paper_composition',WORK/'CollisionPaper/本番フォーマット/prepare_figures.py')
    paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper);paper.register_label_font()
    paper.combine([panels/'size.pdf',panels/'particle_count_lwc_scene_summary.pdf'],FIGURES/'fig04.pdf',2,1.0,
        subtitles=['Number-density distribution','Droplet count and liquid water content'],column_widths=[300.,500.])
    plt.rcdefaults()
    population_velocity_figure()
    case_figure()
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet')
    plt.rcdefaults();review_scores.plot(labels,panels)
    for suffix in ['pdf','png']:
        shutil.copy2(panels/f'review_tp_fp_score_distributions_260715.{suffix}',FIGURES/f'fig11.{suffix}')
    from figure12_thresholds import main as figure12
    figure12()
    events=pd.read_parquet(PRIVATE/'physical_events_all.parquet')
    binary=events.loc[events.status.ne('ambiguous')]
    physical_plot.FIGURE_ROOT=panels
    physical_plot.combined_scatter_figure(binary.loc[binary.diameter_available],binary.loc[binary.velocity_available])
    for suffix in ['pdf','png']:
        shutil.copy2(panels/f'physical_consistency_combined_260715.{suffix}',FIGURES/f'fig13.{suffix}')
    scenes=pd.read_csv(DATA/'scene_collision_counts.csv')
    gravity.save_scene_comparison(FIGURES/'fig14',scenes.iloc[:-1],scenes.iloc[-1])
    pairs=pd.read_csv(DATA/'diameter_pair_counts.csv')
    obs=np.zeros((11,11));pred=np.full((11,11),np.nan)
    for r in pairs.itertuples():obs[r.small_bin_index,r.large_bin_index]=r.confirmed_collision_lower_bound;pred[r.small_bin_index,r.large_bin_index]=r.predicted_expected_count
    gravity.save_pair_map(FIGURES/'fig15',obs,pred,np.sqrt(pred),np.arange(25.,325.,25.))
    writer=PdfWriter()
    for n in [4,5,10,11,12,13,14,15]:writer.append(FIGURES/f'fig{n:02d}.pdf',outline_item=f'Figure {n}')
    writer.write(PACKAGE/'experimental_figures.pdf')
    save(ROOT/'plot-receipt.json',{'created_at':stamp(),'matplotlib':matplotlib.__version__,
        'figures':[4,5,10,11,12,13,14,15],'pdf_fonttype':42})
    print('Created all 8 replacement figures and the combined PDF.')

if __name__=='__main__':main()
