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
    """Use the author's confirmed 03/08 pair and enlarged image panels."""
    from figure10_selected import main as selected
    selected(rebuild_collection=False)


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
