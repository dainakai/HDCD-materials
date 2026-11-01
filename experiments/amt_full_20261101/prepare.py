"""Refit upstream selection and rebuild independent candidate feature tables."""
from common import *
import argparse
import gc
import fcntl
import time
import torch
from threadpoolctl import threadpool_limits
from ra3d.config import KEY_COLUMNS,STAGE1_FEATURES,STATIC_FEATURES,IMAGE_FEATURES,Units
from ra3d.tracks import read_tracks
from ra3d.features import generate_candidates,static_features,image_features
from ra3d.cascade import selected_gate,shortlist,score_static,image_mask,scoring_union
from ra3d.training import fit_classifier,TrainingCheckpoint
from ra3d._core.features import feature_frame,add_derived_features
from ra3d._core import two_d_common as two_d
from ra3d._core.two_d_budget import budget_select
from ra3d.baseline2d import select_union
from ra3d.acquisition import Acquisition,Reconstructor
from ra3d.minip import MinIPReader
from ra3d.io import legacy_candidate_csv


def pool(seed,method):
    if method=='ours':
        source=tracking(seed,method)/'raw.parquet'
        wide=generate_candidates(read_tracks(source))
        write_table(wide,wide_path(seed,method))
        detail='Fresh physical-Z candidates from newly measured phase-3 tracks of the 102.3 mm sequence.'
    else:
        source=tracking(seed,method)/'raw.parquet'
        raw=read_tracks(source).copy()
        raw['z']=1.;raw['vz']=0.;raw['postOutProb']=0.
        # Both methods generate hypotheses from raw observations. Smoothed
        # trajectories are used only for their subsequent temporal features.
        wide=generate_candidates(raw)
        end=raw.groupby('track_id',sort=True).tail(1).set_index('track_id')
        edge=np.minimum.reduce([end.x.to_numpy(),end.y.to_numpy(),1024.-end.x,1024.-end.y])
        boundary=np.maximum(np.maximum(0.,(5.-edge)/5.),
                            np.maximum(0.,(5.-(int(raw.frame.max())-end.frame.to_numpy()))/5.))
        wide['boundary_exit_score']=wide.vanished_track_id.map(pd.Series(boundary,index=end.index))
        for col in ['z_um','vanished_z_um','survivor_z_um']:wide[col]=0.
        wide=two_d.add_2d_derived_features(wide)
        write_table(wide,path(seed,method,'wide.parquet'))
        detail='Independent raw-track XY candidate generation, no depth and no legacy reference-bank import.'
    gate=selected_gate(wide) if method=='ours' else two_d.selected_gate_mask(wide)
    audit={'seed':seed,'method':method,'wide_rows':len(wide),'gate_rows':int(gate.sum()),
           'source':str(source),'sha256':sha256(source),'candidate_definition':detail}
    if seed in TRAIN:
        labeled,info=labels(wide.loc[gate].reset_index(drop=True),seed,method)
        if method!='ours':
            ref=read_table(path(seed,'ours','stage1_training.parquet'))
            budget=int((ref.spatial_positive.eq(0)|ref.identity_training_selected).sum())
            usable=labeled.loc[labeled.spatial_positive.eq(0)|labeled.identity_training_selected]
            positive=usable.loc[usable.identity_training_selected]
            negative=usable.loc[~usable.identity_training_selected]
            required=budget-len(positive)
            if not 0<required<=len(negative):raise ValueError(f'Invalid matched training budget {required}')
            hashes=pd.util.hash_pandas_object(negative[KEY_COLUMNS],index=False).to_numpy(np.uint64)
            order=np.lexsort((negative.survivor_track_id,negative.vanished_track_id,negative.candidate_frame,hashes))
            labeled=pd.concat([positive,negative.iloc[order[:required]]],ignore_index=True)
            audit['matched_training_budget']=budget
        write_table(labeled,path(seed,method,'stage1_training.parquet' if method=='ours' else 'gate.parquet'))
        audit['labels']=info
    write_json(audit,path(seed,method,'pool.json'));emit(stage='pool_complete',**audit)


def upstream(seed,method,stage):
    train=[s for s in TRAIN if s!=seed]
    filename=({'stage1':'stage1_training.parquet','static':'static77.parquet'} if method=='ours'
              else {'stage1':'gate.parquet','static':'static.parquet'})[stage]
    columns=({'stage1':STAGE1_FEATURES,'static':STATIC_FEATURES} if method=='ours'
             else {'stage1':list(two_d.STAGE1_FEATURES),'static':list(two_d.TWO_D_FEATURES)})[stage]
    inputs=[path(s,method,filename) for s in train]
    identity={'protocol':sha256(ROOT/'config/protocol.json'),'engine':sha256(SCRIPT/'engine-manifest.json'),'held_seed':seed,'training_seeds':train,
              'features':columns,'source_files':{str(p):sha256(p) for p in inputs}}
    check=TrainingCheckpoint(ROOT/f'models/upstream/{method}/seed{seed}/{stage}',identity)
    model,audit=check.tree('model',lambda:fit_classifier((read_table(p) for p in inputs),columns),lambda v:emit(**v))
    return model,identity


def static(seed,method):
    model,identity=upstream(seed,method,'stage1')
    wide=read_table(wide_path(seed,method))
    columns=STAGE1_FEATURES if method=='ours' else list(two_d.STAGE1_FEATURES)
    scores=model.predict_proba(feature_frame(wide,columns))[:,1]
    if method=='ours':
        short=shortlist(wide,scores);selection={'outside_fraction':.4}
    else:
        import pyarrow.parquet as pq
        budget=pq.ParquetFile(path(seed,'ours','shortlist.parquet')).metadata.num_rows
        short,selection=budget_select(wide,scores,budget)
    del wide,model,scores;gc.collect()
    short=legacy_candidate_csv(short)
    write_table(short,path(seed,method,'shortlist.parquet'))
    smooth=read_tracks(tracking(seed,method)/'smoothed.parquet')
    if method=='ours':
        frame=static_features(short,smooth,tracking(seed,method)/'smoothing/smoothedtrajs',Units(),lambda v:emit(seed=seed,method=method,**v))
        selection['model_independent_cached_rows']=0
    else:
        smooth=smooth.copy();smooth['z']=0.;smooth['vz']=0.;smooth['postOutProb']=0.
        frame=static_features(short,smooth,None,Units(),lambda v:emit(seed=seed,method=method,**v))
        frame=two_d.add_2d_derived_features(frame)
    if seed in TRAIN:frame,label_audit=labels(frame,seed,method)
    else:frame['synth_seed']=seed;label_audit=None
    write_table(frame,path(seed,method,'static77.parquet' if method=='ours' else 'static.parquet'))
    write_json({'status':'PASS','seed':seed,'method':method,'rows':len(frame),'upstream':identity,
                'selection':selection,'labels':label_audit},path(seed,method,'static.json'))
    emit(stage='static_complete',seed=seed,method=method,rows=len(frame))


def score(seed,method):
    model,identity=upstream(seed,method,'static')
    frame=read_table(path(seed,method,'static77.parquet' if method=='ours' else 'static.parquet'))
    if method=='ours':
        scored=score_static(frame,model)
        requested=scored.loc[image_mask(scored)].reset_index(drop=True)
        write_table(scored,path(seed,method,'static_scored.parquet'))
    else:
        full=select_union(frame,model)
        requested=full.loc[full.static_global_top25.eq(1)].reset_index(drop=True)
        write_table(full,path(seed,method,'full.parquet'))
    write_table(requested,path(seed,method,'image_requested.parquet'))
    write_json({'status':'PASS','seed':seed,'method':method,'upstream':identity,'requested':len(requested)},path(seed,method,'score.json'))
    emit(stage='score_complete',seed=seed,method=method,requested=len(requested))


def images(seed,method,device):
    requested=read_table(path(seed,method,'image_requested.parquet'))
    tracks=read_tracks(tracking(seed,method)/'smoothed.parquet')
    if method=='ours':
        if device.startswith('cuda'):assert torch.cuda.is_available()
        recon=Reconstructor(Acquisition.load(ROOT/f'config/seed{seed}-collision.yaml'),device)
        image=image_features(requested,tracks,recon,progress=lambda v:emit(seed=seed,method=method,**v),checkpoint_path=path(seed,method,'image_checkpoint.npz'))
        scored=read_table(path(seed,method,'static_scored.parquet'))
        full=scoring_union(scored,image)
    else:
        tracks=tracks.copy();tracks['z']=0.;tracks['vz']=0.;tracks['postOutProb']=0.
        recon=MinIPReader(minip_dir(seed),'depth-free' if method=='depth_free' else 'minip-2d')
        image=image_features(requested,tracks,recon,progress=lambda v:emit(seed=seed,method=method,**v),checkpoint_path=path(seed,method,'image_checkpoint.npz'))
        full=read_table(path(seed,method,'full.parquet'))
        full=full.drop(columns=IMAGE_FEATURES,errors='ignore').merge(image[KEY_COLUMNS+IMAGE_FEATURES],on=KEY_COLUMNS,how='left',sort=False,validate='one_to_one')
        full=add_derived_features(full);full['image_enriched']=full.static_global_top25.astype(np.int8)
    if seed in TRAIN:full,label_audit=labels(full,seed,method)
    else:full['synth_seed']=seed;label_audit=None
    write_table(image[KEY_COLUMNS+IMAGE_FEATURES],path(seed,method,'image_features.parquet'))
    write_table(full,path(seed,method,'paper92.parquet' if method=='ours' else 'full64.parquet'))
    write_json({'status':'PASS','seed':seed,'method':method,'requested':len(requested),'full_rows':len(full),'labels':label_audit,'device':device},path(seed,method,'images.json'))
    emit(stage='images_complete',seed=seed,method=method,full_rows=len(full))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pool','static','score','images'])
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--method',choices=['ours','depth_free','minip2d'],required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads)
    # A second GPU can take an unstarted scene without risking duplicate writes.
    lock=path(a.seed,a.method,a.stage+'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if path(a.seed,a.method,a.stage+'.json').exists():emit(status='ALREADY_COMPLETE',**vars(a));raise SystemExit(0)
    with threadpool_limits(limits=a.threads):
        started=time.monotonic()
        if a.stage=='images':images(a.seed,a.method,a.device)
        else:globals()[a.stage](a.seed,a.method)
        emit(status='PASS',seconds=time.monotonic()-started,**vars(a))
