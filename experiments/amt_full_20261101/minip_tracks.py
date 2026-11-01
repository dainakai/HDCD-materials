"""Fresh XY-only linking of the common 3-iteration endpoint-inclusive MinIP measurements."""
from common import *
import argparse
import gc
import shutil
import time
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from holod3_linker.runner import link_holod3_output
from holod3_smoother.core import smooth_xy
from holod3_smoother.config import SmootherConfig
from ra3d.tracks import read_tracks

def match_identity(raw,truth):
    truth=truth.loc[truth.center_in_volume.eq(1)].sort_values(['frame','particle_id'],kind='stable')
    groups=truth.groupby('frame',sort=False).indices
    rows=[]
    for frame,part in raw.sort_values(['frame','track_id'],kind='stable').groupby('frame',sort=True):
        true=truth.iloc[groups[int(frame)]]
        dx=part.x.to_numpy()[None,:]-true.x_px.to_numpy()[:,None]
        dy=part.y.to_numpy()[None,:]-true.y_px.to_numpy()[:,None]
        cost=np.hypot(dx/8.,dy/8.)
        cost[(np.abs(dx)>8.)|(np.abs(dy)>8.)]=1e9
        ti,pi=linear_sum_assignment(cost);keep=cost[ti,pi]<=1.
        ti,pi=ti[keep],pi[keep]
        rows.append(pd.DataFrame({'frame':int(frame),'observed_track_id':part.track_id.to_numpy()[pi],
                                 'source_particle_id':true.particle_id.to_numpy()[ti],'match_cost':cost[ti,pi]}))
        if (frame+1)%1000==0:emit(stage='identity_mapping',frame=int(frame))
    out=pd.concat(rows,ignore_index=True).sort_values(['frame','observed_track_id'],kind='stable')
    assert not out.duplicated(['frame','observed_track_id']).any()
    assert not out.duplicated(['frame','source_particle_id']).any()
    return out

def run(seed):
    started=time.monotonic();out=tracking(seed,'minip2d');out.mkdir(parents=True,exist_ok=True)
    marker=out/'complete.json'
    if marker.exists():emit(status='ALREADY_COMPLETE',seed=seed);return
    source=measurement(seed)/'particles_3d.csv'
    fields=list(pd.read_csv(source,nrows=0).columns)
    fields=fields[:fields.index('seg_area_px')+1]
    assert not {'slice','depth_um','z_um','final_diameter_um'} & set(fields)
    inputs=out/'input';inputs.mkdir(exist_ok=True);csvpath=inputs/'particles_3d.csv'
    if not csvpath.exists():
        # Parse only the MinIP ROI prefix. Estimated depth and focused diameter
        # columns are neither read into the frame nor used by this method.
        frame=pd.read_csv(source,usecols=fields)
        frame['frame']=frame.file.map(lambda v:int(Path(v).stem)-1).astype('int64')
        assert frame.frame.between(0,5999).all()
        frame['slice']=1.;frame['z_um']=0.
        frame['final_diameter_um']=frame.diameter_um
        frame['final_diameter_source']='minip_contrast_area'
        write_table(frame,csvpath)
        write_json({'source':str(source),'source_sha256':sha256(source),'read_columns':fields,
                    'rows':len(frame),'input_sha256':sha256(csvpath),
                    'depth':'Constant z_um=0; slice=1 is only an adapter schema placeholder.',
                    'image_source':'74.9–177.2 mm, 3 phase iterations, 1536-pixel padding, 513 MinIP planes'},out/'input.json')
        del frame;gc.collect()
    linked=out/'linking'
    if not (linked/'holod3_link_summary.json').exists():
        if linked.exists():shutil.rmtree(linked) # Only incomplete outputs owned by this seed/job.
        emit(stage='link_start',seed=seed)
        link_holod3_output(csvpath,output_dir=linked,pixel_pitch_um=10.,z_weight=0.,z_tiebreak_penalty=0.)
    if not (out/'smoothed.parquet').exists():
        raw=read_tracks(linked/'trajectory3d')
        # Exported slice can preserve the adapter's one-based placeholder.
        # Both candidates and features explicitly use the zero-depth schema.
        raw['z']=0.
        smooth=raw.copy()
        for col in ['vx','vy','vz','postOutProb']:smooth[col]=0.
        config=SmootherConfig(xy_mode='legacy')
        for tid,part in raw.groupby('track_id',sort=False):
            assert np.all(np.diff(part.frame)>0)
            for col in 'xy':
                position,velocity=smooth_xy(part[col].to_numpy(),part.frame.to_numpy(),config)
                smooth.loc[part.index,col]=position;smooth.loc[part.index,'v'+col]=velocity
        write_table(raw,out/'raw.parquet');write_table(smooth,out/'smoothed.parquet')
        emit(stage='smoothing_complete',seed=seed,rows=len(raw),tracks=int(raw.track_id.nunique()))
    else:raw=read_tracks(out/'raw.parquet')
    truthfile=truth_path(seed,'particles.parquet')
    if not (out/'mapping.parquet').exists():
        truth=pd.read_parquet(truthfile,columns=['frame','particle_id','x_px','y_px','center_in_volume'])
        write_table(match_identity(raw,truth),out/'mapping.parquet')
    diagnostic=json.loads((linked/'depth3d_link_diagnostics.json').read_text())
    assert diagnostic['z_weight']==0.
    write_json({'status':'PASS','seed':seed,'rows':len(raw),'tracks':int(raw.track_id.nunique()),
                'first_frame':int(raw.frame.min()),'last_frame':int(raw.frame.max()),
                'input':json.loads((out/'input.json').read_text()),
                'truth_sha256':sha256(truthfile),
                'parameters':json.loads((linked/'holod3_link_summary.json').read_text())['parameters'],
                'outputs':{n:sha256(out/n) for n in ['raw.parquet','smoothed.parquet','mapping.parquet']},
                'seconds':time.monotonic()-started},marker)
    # Retain compact, auditable tables and diagnostics; discard redundant CSVs.
    for name in ['depth3d_link_diagnostics.json','holod3_link_summary.json']:
        shutil.copy2(linked/name,out/name)
    shutil.rmtree(linked);shutil.rmtree(inputs)
    emit(stage='complete',seed=seed,seconds=time.monotonic()-started)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(args.seed)
