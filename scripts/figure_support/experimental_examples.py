"""Trajectories and five-frame views of two confirmed collisions."""
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
from pathlib import Path
import hashlib
INPUT = Path(__file__).resolve().parents[2] / "data/experimental"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

SELECTED=[('a','03','15:6509:6639:6750'),('b','08','22:3758:4041:4087')]
COLORS=['#d94235','#2674b9']
WIDTH_IN=12.8
VIEW_PIXELS={'03':96,'08':120}


def view_bounds(case):
    width=VIEW_PIXELS[case['option']]
    return 63.5-width/2,63.5+width/2


def image_panel(ax,case,rec,groups,scale=False,standalone=False):
    data=np.load(INPUT/'figure10'/rec['cache'])
    assert sha(INPUT/'figure10'/rec['cache'])==rec['sha256']
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



def main(output):
    import json
    cases=json.loads((INPUT/'figure10/cases.json').read_text())
    tracks=pd.read_parquet(INPUT/'collisions/trajectories.parquet')
    groups={c['candidate_id']: [g.sort_values('frame') for _,g in
        tracks.loc[tracks.candidate_id.eq(c['candidate_id'])].groupby('track_id',sort=True)] for c in cases}
    labels=pd.read_csv(INPUT/'reviews.csv').set_index('candidate_id')
    for c in cases:
        row=labels.loc[c['candidate_id']]
        local=tracks.loc[tracks.candidate_id.eq(c['candidate_id'])]
        groups[c['candidate_id']]=[local.loc[local.track_id.eq(t)].sort_values('frame') for t in [row.vanished_track_id,row.survivor_track_id]]
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':11})
    return plate(cases,groups,output/'fig10')
