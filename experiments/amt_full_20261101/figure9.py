#!/usr/bin/env python3
"""Rebuild the illustrative image-feature examples with new track identities and scores."""
from pathlib import Path
import sys
import json
# Resolve the pinned optional dependencies before pandas caches Arrow support.
from common import ROOT,path,tracking,events
import numpy as np
import pandas as pd
import torch
from ra3d.strict_labels import attach_identity_labels as label_candidates
from ra3d.acquisition import Acquisition,Reconstructor
from ra3d.io import write_json,write_table
from ra3d.config import KEY_COLUMNS
from ra3d.tracks import read_tracks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def style():
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    output=ROOT/'figures';output.mkdir(exist_ok=True)
    fig.savefig(output/(name+'.png'),dpi=170)
    fig.savefig(output/(name+'.pdf'))
def numerical_tracks(seed,_):return read_tracks(tracking(seed,'ours')/'smoothed.parquet')



def main():
    torch.set_num_threads(4);style()
    full=pd.read_parquet(path(260745,'ours','paper92.parquet'))
    scores=pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet')
    full=full.merge(scores,on=KEY_COLUMNS,validate='one_to_one')
    mapping=pd.read_parquet(tracking(260745,'ours')/'mapping.parquet')
    truth=events(260745)
    labeled,audit,_=label_candidates(full,truth,mapping,z_tolerance_um=15000.)
    # Training labels populate identity metadata only for spatial positives.
    # Join the same saved frame/track associations for negative illustrations too;
    # this leaves every training label, fitted model and test score unchanged.
    lookup=mapping.set_index(['frame','observed_track_id']).source_particle_id
    for role,frame_column,track_column in [
            ('vanished','vanished_end_frame','vanished_track_id'),
            ('survivor','candidate_frame','survivor_track_id')]:
        keys=pd.MultiIndex.from_arrays([labeled[frame_column],labeled[track_column]],
                                      names=['frame','observed_track_id'])
        labeled[f'{role}_source_particle_id']=lookup.reindex(keys).to_numpy()
    audit['illustration_identity_metadata']='For all candidates, including spatial negatives, join the saved XY truth mapping at the terminating-track end frame and continuing-track candidate frame. This is illustration metadata only.'
    tracks=numerical_tracks(260745,True)
    life=tracks.groupby('track_id').frame.agg(['min','max'])
    arrays={int(k):v.sort_values('frame')[['frame','x','y','z','diameter_px']].to_numpy(float)
            for k,v in tracks.groupby('track_id',sort=False)}

    def visible_pair(row):
        """Fixed crops must stay inside the image and show both pre-event tracks."""
        t=int(row.candidate_frame)
        v=arrays[int(row.vanished_track_id)];s=arrays[int(row.survivor_track_id)]
        if not v[0,0]<=t-4<=v[-1,0]:return False
        for frame in (t-4,t,t+4):
            sx=np.interp(frame,s[:,0],s[:,1]);sy=np.interp(frame,s[:,0],s[:,2])
            if not (72<=sx<=952 and 72<=sy<=952):return False
            if frame==t-4:
                vx=np.interp(frame,v[:,0],v[:,1]);vy=np.interp(frame,v[:,0],v[:,2])
                vr=max(4.,np.interp(frame,v[:,0],v[:,4])/2.)
                if abs(vx-sx)+vr>66 or abs(vy-sy)+vr>66:return False
        return True
    selected=labeled.loc[labeled.image_enriched.eq(1)&labeled.rendered_pre_frames.ge(7)].copy()
    selected=selected.loc[(selected.candidate_frame-4>=selected.survivor_track_id.map(life['min'])) &
                          (selected.candidate_frame+4<=selected.survivor_track_id.map(life['max'])) &
                          (selected.candidate_frame-4>=selected.vanished_track_id.map(life['min']))]
    true=selected.loc[selected.identity_state.eq('identity_consistent')].copy()
    true_frame=truth.set_index('collision_event_id').frame
    true=true.loc[(true.candidate_frame-true.matched_truth_event_id.map(true_frame)).abs().le(1)]
    true['joint_score']=np.minimum(true.static_score,true.base_score)
    true=true.sort_values(['joint_score','candidate_frame','vanished_track_id','survivor_track_id'],
                         ascending=[False,True,True,True],kind='stable')
    chosen_true=next((row for _,row in true.iterrows() if visible_pair(row)),None)
    if chosen_true is None:raise RuntimeError('No identity-consistent collision has the required visible display window.')
    artifact=selected.loc[selected.vanished_source_particle_id.eq(selected.survivor_source_particle_id)&
                           selected.vanished_source_particle_id.notna()&selected.spatial_positive.eq(0)&
                           selected.pre_same_interp_fraction.ge(7/9)].copy()
    artifact['score_decrease']=artifact.static_score-artifact.base_score
    artifact=artifact.sort_values(['score_decrease','static_score','candidate_frame'],
                                  ascending=[False,False,True],kind='stable')
    chosen_artifact=None;agreement=0;role_support=None
    for _,row in artifact.iterrows():
        if not visible_pair(row):continue
        support=[0,0];consistent_frames=0;conflicts=0
        source=int(row.vanished_source_particle_id)
        for f in range(int(row.candidate_frame)-10,int(row.candidate_frame)-1):
            v=lookup.get((f,int(row.vanished_track_id)),np.nan)
            s=lookup.get((f,int(row.survivor_track_id)),np.nan)
            # The truth association is one-to-one within a frame. A shared-particle
            # artifact therefore has alternating support across track IDs, not
            # simultaneous assignments of the same truth ID to both tracks.
            support[0]+=int(np.isfinite(v) and v==source)
            support[1]+=int(np.isfinite(s) and s==source)
            consistent_frames+=int(v==source or s==source)
            conflicts+=int((np.isfinite(v) and v!=source) or (np.isfinite(s) and s!=source))
        if consistent_frames>=5 and min(support)>=2 and conflicts==0:
            chosen_artifact=row;agreement=consistent_frames;role_support=support;break
    if chosen_artifact is None:
        raise RuntimeError('No verified shared-particle example satisfies the documented display rule; report this instead of inventing one.')
    cases=[('Ground-truth collision',chosen_true),('Shared-particle artifact',chosen_artifact)]
    recon=Reconstructor(Acquisition.load(ROOT/'config/seed260745-collision.yaml'),'cuda:0')
    fig=plt.figure(figsize=(13,7),layout='constrained')
    grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,1.3])
    colors=['#D84A3A','#2C6FB7'];records=[]
    for r,(title,row) in enumerate(cases):
        t=int(row.candidate_frame);ids=[int(row.vanished_track_id),int(row.survivor_track_id)]
        parts=[tracks.loc[tracks.track_id.eq(i)].sort_values('frame') for i in ids]
        for c,delta in enumerate([-4,0,4]):
            f=t+delta;sv=parts[1]
            sx=np.interp(f,sv.frame,sv.x);sy=np.interp(f,sv.frame,sv.y);sz=np.interp(f,sv.frame,sv.z)
            plane=recon.intensity(f,sz)
            # Fixed field of view, tracked continuing-particle focus; no manual refocus.
            side=144;left=int(round(sx))-side//2;top=int(round(sy))-side//2
            image=np.full((side,side),float(np.mean(plane)),dtype=float)
            x0,x1=max(0,left),min(plane.shape[1],left+side)
            y0,y1=max(0,top),min(plane.shape[0],top+side)
            image[y0-top:y1-top,x0-left:x1-left]=plane[y0:y1,x0:x1]
            ax=fig.add_subplot(grid[r,c]);ax.imshow(image,cmap='gray',vmin=0,vmax=1)
            for tr,color in zip(parts,colors):
                if tr.frame.min()<=f<=tr.frame.max():
                    x=np.interp(f,tr.frame,tr.x)-left;y=np.interp(f,tr.frame,tr.y)-top
                    rad=max(4.,np.interp(f,tr.frame,tr.diameter_px)/2.)
                    ax.add_patch(plt.Circle((x,y),rad,fill=False,color=color,lw=1.3))
            ax.plot([8,58],[side-13,side-13],color='white',lw=3)
            ax.text(33,side-18,'500 µm',ha='center',color='white',fontsize=9,
                     bbox={'facecolor':'black','edgecolor':'none','alpha':.45,'pad':1})
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{delta/4:+.1f} ms',fontsize=11)
            if c==0:ax.set_ylabel(f'({chr(97+r)}) {title}',fontsize=11,labelpad=10)
        ax=fig.add_subplot(grid[r,3])
        values=[float(row.static_score),float(row.base_score)]
        ax.bar([0,1],values,color=['#79838A','#009E73'],width=.6)
        for x,v in enumerate(values):ax.text(x,min(v+.025,1.07),f'{v:.3f}',ha='center',fontsize=12)
        ax.set_xticks([0,1],['Trajectory\nclassifier','Image-feature\nensemble']);ax.set_ylim(0,1.12)
        ax.set_ylabel('Candidate score');ax.set_title('Scores before relation correction',fontsize=11)
        records.append({'case':title,'candidate_frame':t,'vanished_track_id':ids[0],'survivor_track_id':ids[1],
                         'true_event_id':int(row.matched_truth_event_id),
                         'vanished_truth_id':int(row.vanished_source_particle_id),
                         'survivor_truth_id':int(row.survivor_source_particle_id),
                         'trajectory_score':values[0],'image_feature_score':values[1],
                         'same_truth_particle_preframes':None if r==0 else agreement,
                         'terminating_truth_support_preframes':None if r==0 else role_support[0],
                         'continuing_truth_support_preframes':None if r==0 else role_support[1],
                         'image_pre_same_fraction':float(row.pre_same_interp_fraction)})
    fig.suptitle('Figure 9. Newly trained model with aligned conditions and adjacent IDs\n'
                 'Red: terminating track; blue: continuing track. Times relative to candidate frame.',fontsize=14)
    save(fig,'figure09');plt.close(fig)
    write_table(pd.DataFrame(records),ROOT/'results/figure9_cases.csv')
    write_json({'status':'PASS','cases':records,'selection':
                 'Select the highest joint-score identity-consistent event and the largest trajectory-to-image score decrease among eligible shared-particle artifacts. For visibility, all 144-pixel crops must lie inside the image, both tracks must fit inside the pre-event crop, and the terminating track must still exist four frames before the candidate. The true-event frame is within one frame of the candidate. The artifact has a shared-component fraction of at least 7/9 and, in nine preceding frames, at least five distinct frames supporting the same truth particle, at least two mapped observations from each track ID, and no different truth identity. The visibility criteria were fixed using the previous version before examining this new dataset. Simultaneous duplicate truth assignments are impossible under the one-to-one mapping. Selection affects illustration only; no fit or performance threshold is changed.',
                 'focus':'continuing-track smoothed depth at each displayed frame; no manual refocus',
                 'label_audit':audit},ROOT/'results/figure9.json')


if __name__=='__main__':main()
