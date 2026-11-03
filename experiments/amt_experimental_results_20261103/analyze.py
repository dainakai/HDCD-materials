"""Recompute experimental diagnostics from the frozen reviews and new measurements."""
import sys
import math
import numpy as np
import pandas as pd
import polars as pl
from common import *
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'src'))
from figure_support import review_rank, gravity as g, population_size, population_count
from ra3d.physics import consistency, collision_rate
from ra3d._core.physics import median_velocity_px_per_frame

def metrics(frame, expected, measured):
    x, y = frame[expected].to_numpy(float), frame[measured].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    e = y - x
    return dict(n=len(x), bias=float(e.mean()), mae=float(np.abs(e).mean()),
        median_absolute_error=float(np.median(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
        correlation=float(np.corrcoef(x, y)[0,1]))

def review_tables(labels):
    top = review_rank.validate_top_score_cohort(labels)
    curve = review_rank.aggregate_by_rank(top)
    total = int(labels.status.eq('collision').sum())
    curve['coverage_of_reviewed_confirmed'] = curve.cumulative_confirmed_collisions / total
    curve['binary_only_precision'] = curve.cumulative_confirmed_collisions / (curve.cumulative_confirmed_collisions + curve.cumulative_non_collisions)
    curve.to_csv(DATA / 'review_budget.csv', index=False)
    scene = pd.crosstab(labels.scene_num, labels.status).reindex(columns=review_rank.STATUS_ORDER).reset_index()
    scene.insert(0,'scene_index',range(1,15))
    scene['reviewed'] = 80
    scene.to_csv(DATA / 'review_counts_by_scene.csv',index=False)
    columns = ['scene_num','review_score_rank','collision_score','status']
    labels[columns].to_csv(PRIVATE / 'review_labels_for_verification.csv', index=False)
    n95 = int(curve.loc[curve.coverage_of_reviewed_confirmed.ge(.95),'top_n_per_scene'].iloc[0])
    result = {'reviewed':len(labels),'confirmed':total,'non_collision':int(labels.status.eq('non_collision').sum()),
        'ambiguous':int(labels.status.eq('ambiguous').sum()),'precision_all_reviewed':total/len(labels),
        'precision_binary':total/int(labels.status.ne('ambiguous').sum()),'minimum_n_for_95pct':n95,
        'rounded_n_for_95pct':int(math.ceil(n95/10)*10),
        'top30':curve.loc[curve.top_n_per_scene.eq(30)].iloc[0].to_dict(),
        'score_min':float(labels.score.min()),'score_max':float(labels.score.max())}
    return result

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    frozen = read(PRIVATE / 'freeze.json')
    assert sha(PRIVATE / 'reviewed_candidates.parquet') == frozen['joined_sha256']
    labels = pd.read_parquet(PRIVATE / 'reviewed_candidates.parquet')
    summary = {'review':review_tables(labels),'conventions':dict(frames_per_scene=FRAMES,fps=FPS,volume_cm3=VOLUME_CM3,
        exposure_per_scene_m3_s=EXPOSURE,pooled_exposure_m3_s=EXPOSURE*14,uncertainty='conditional Poisson SD = sqrt(expected count)',
        density_bins='5 um, 25 <= D <= 300 um (numpy histogram final edge inclusive)',pair_bins='25 um, 25 <= D < 300 um',
        velocity='median observed y-displacement / frame interval within diameter-derived windows; minimum 3 points',
        diameter='median of last 10 observations at or before f-2 and first 20 after f; finite positive diameters',
        source_model_sha256=MODEL_SHA,review_snapshot_sha256=frozen['snapshot_sha256']), 'scene_audits':[]}
    fine_edges=np.arange(25.,305.,5.)
    centers=(fine_edges[:-1]+fine_edges[1:])/2
    pair_edges=np.arange(25.,325.,25.)
    kernel=g.HallCorrectedLaminarKernel(REPO/'src/ra3d/config/physics')
    a,b=np.meshgrid(centers,centers,indexing='ij')
    matrix=kernel.kernel_m3_s(a,b)
    physical=[]; densities=[]; frames_all=[]; tracks_all=[]; scenes=[]; pooled=np.zeros((11,11)); source_checks=[]
    for idx, scene in enumerate(SCENES,1):
        root=CAMPAIGN/f'scenes/scene{scene:04d}'
        tracks=pd.read_parquet(root/'tracking/smoothed.parquet')
        selected=labels.loc[labels.scene_num.eq(scene)].copy()
        result=consistency(selected,tracks)
        # These are the exact diameter features used during inference, recomputed independently.
        for column in ['pre_diameter_vanished_um','pre_diameter_survivor_um','post_diameter_um',
                       'pre_window_start_frame','pre_window_end_frame','post_window_start_frame','post_window_end_frame']:
            np.testing.assert_allclose(result[column],selected[column],rtol=1e-12,atol=1e-10,equal_nan=True)
        groups={int(k):v for k,v in tracks.groupby('track_id',sort=False)}
        velocities=[]
        for row in result.itertuples():
            vals=[]
            for tid,when in [(row.vanished_track_id,'pre'),(row.survivor_track_id,'pre'),(row.survivor_track_id,'post')]:
                start,end=getattr(row,f'{when}_window_start_frame'),getattr(row,f'{when}_window_end_frame')
                value,count=median_velocity_px_per_frame(groups[int(tid)],int(start),int(end)) if np.isfinite([start,end]).all() else (np.nan,0)
                vals.append(value*.04)
            velocities.append(vals)
        result[['pre_vy_vanished_m_s','pre_vy_survivor_m_s','post_vy_unfiltered_m_s']]=velocities
        result['diameter_available']=np.isfinite(result[['expected_post_diameter_um','measured_post_diameter_um']]).all(axis=1)&result.measured_post_diameter_um.gt(0)
        result['classification_role']=result.status.map({'collision':'TP','non_collision':'FP','ambiguous':'ambiguous'})
        physical.append(result)
        # The frame-wise particle table, before trajectory filtering, defines number density.
        particles=pl.read_csv(root/'measurement/particles_3d.csv',columns=['frame','final_diameter_um'],n_threads=4).to_pandas()
        assert particles.frame.between(0,FRAMES-1).all()
        good=np.isfinite(particles.final_diameter_um)&particles.final_diameter_um.gt(0)
        assert good.all(), 'Invalid measurements require explicit treatment.'
        d=particles.final_diameter_um.to_numpy(float)
        counts=np.histogram(d,bins=fine_edges)[0]
        density=counts/(FRAMES*VOLUME_CM3)
        pred=g.pair_prediction(density*1e6,matrix,centers,pair_edges,exposure_volume_m3_s=EXPOSURE)[0]
        # Independently exposed public physics API must agree on every pair cell and pooled count.
        check,audit=collision_rate(particles.rename(columns={'final_diameter_um':'diameter_um'}),recorded_frames=FRAMES,volume_cm3=VOLUME_CM3)
        np.testing.assert_allclose(check.predicted_count.to_numpy(),pred[np.triu_indices(11)],rtol=1e-12,atol=1e-12)
        # A full symmetric fine-bin integration is a second check of unordered-pair normalization.
        direct=EXPOSURE*.5*np.einsum('i,ij,j',density*1e6,matrix,density*1e6)
        np.testing.assert_allclose(direct,np.nansum(pred),rtol=1e-12)
        pooled+=np.nan_to_num(pred)
        expected=float(np.nansum(pred)); observed=int(selected.status.eq('collision').sum())
        scenes.append(dict(row_type='scene',scene_index=idx,raw_scene_num=scene,display_label=f'Scene {idx}',
            confirmed_collision_lower_bound=observed,predicted_expected_count=expected,exposure_volume_m3_s=EXPOSURE))
        for lo,hi,c,count,density_value in zip(fine_edges[:-1],fine_edges[1:],centers,counts,density):
            densities.append(dict(scene_num=scene,diameter_low_um=lo,diameter_high_um=hi,diameter_center_um=c,
                recorded_particle_count=int(count),number_density_cm3=float(density_value),dn_dD_cm3_um=float(density_value/5)))
        frame_index=particles.frame.to_numpy(int)
        count_per_frame=np.bincount(frame_index,minlength=FRAMES)
        cube=np.bincount(frame_index,weights=d**3,minlength=FRAMES)
        # D in um: volume * water density / sampled air volume -> g/m^3.
        lwc=cube*(np.pi/6)*1e-6/VOLUME_CM3
        frames_all.append(pd.DataFrame(dict(scene=f'C000H001S{scene:04d}',scene_number=scene,frame=np.arange(FRAMES),
            elapsed_time_s=np.arange(FRAMES)/FPS,particles_per_hologram=count_per_frame,sum_diameter_cubed_um3=cube,lwc_g_m3=lwc)))
        cols=['diameter_px','vx','vy','vz']
        valid=np.isfinite(tracks[cols]).all(axis=1)&tracks.diameter_px.gt(0)
        track_stat=tracks.loc[valid].groupby('track_id').agg(n_valid_points=('frame','size'),frame_first=('frame','min'),frame_last=('frame','max'),
            diameter_px_median=('diameter_px','median'),vx_px_frame_median=('vx','median'),vy_px_frame_median=('vy','median'),vz_slice_frame_median=('vz','median')).reset_index()
        track_stat['scene']=scene
        track_stat['diameter_um']=track_stat.diameter_px_median*10
        for axis,factor in [('x',.04),('y',.04),('z',.4)]:
            track_stat[f'v{axis}_m_s']=track_stat[f'v{axis}_{"slice" if axis=="z" else "px"}_frame_median']*factor
        tracks_all.append(track_stat.loc[track_stat.n_valid_points.ge(5)&track_stat.diameter_um.between(25,300)])
        summary['scene_audits'].append(dict(scene=scene,particle_rows=len(particles),nonfinite_or_nonpositive=int((~good).sum()),
            below25=int((d<25).sum()),above300=int((d>300).sum()),equal300=int((d==300).sum()),
            fine_histogram_total=int(counts.sum()),measurement_frame_count=int(particles.frame.nunique()),
            unique_track_count=int(tracks.track_id.nunique()),selected_velocity_tracks=len(tracks_all[-1]),
            gravity_independent_api_max_abs_error=float(np.max(abs(check.predicted_count.to_numpy()-pred[np.triu_indices(11)])))))
        print(f'Analyzed scene {scene}: {observed} confirmed, gravity prediction {expected:.4f}',flush=True)
    events=pd.concat(physical,ignore_index=True)
    events.to_parquet(PRIVATE/'physical_events_all.parquet',index=False)
    binary=events.loc[events.status.ne('ambiguous')]
    public_columns=['scene_num','review_rank','classification_role','expected_post_diameter_um','measured_post_diameter_um',
        'expected_post_vy_m_s','measured_post_vy_m_s','diameter_available','velocity_available',
        'pre_vy_vanished_n','pre_vy_survivor_n','post_vy_n']
    binary[public_columns].to_csv(DATA/'physical_consistency.csv',index=False)
    diagnostics=[]
    old=pd.read_csv(REPO/'data/paper_figures/reference/physical_consistency_events.csv')
    for version,frame in [('public',old),('revision',binary)]:
        for role in ['TP','FP']:
            for quantity,expected,measured,valid_col in [('diameter_um','expected_post_diameter_um','measured_post_diameter_um','diameter_available'),
                ('velocity_m_s','expected_post_vy_m_s','measured_post_vy_m_s','velocity_available')]:
                subset=frame.loc[frame.classification_role.eq(role)&frame[valid_col].astype(bool)]
                diagnostics.append(dict(version=version,role=role,quantity=quantity,**metrics(subset,expected,measured)))
    pd.DataFrame(diagnostics).to_csv(DATA/'physical_error_summary.csv',index=False)
    summary['physical']=diagnostics
    pair_labels=labels.copy()
    parent_valid=np.isfinite(pair_labels[['pre_diameter_vanished_um','pre_diameter_survivor_um']]).all(axis=1)&pair_labels.pre_diameter_vanished_um.gt(0)&pair_labels.pre_diameter_survivor_um.gt(0)
    # The feature extractor's min/max use the one available parent when the other is absent.
    # Such a candidate is still a human-confirmed event, but cannot be assigned a diameter pair.
    pair_labels.loc[~parent_valid,['pre_diameter_small_um','pre_diameter_large_um']]=np.nan
    observed,pair_audit=g.observed_collision_counts(pair_labels,pair_edges)
    pair_audit['missing_parent_diameters']=int((~parent_valid&labels.status.eq('collision')).sum())
    assert int(observed.sum())+pair_audit['outside_pair_map']==summary['review']['confirmed'],pair_audit
    summary['pair_map_audit']=pair_audit
    pairs=[]
    for i,j in zip(*np.triu_indices(11)):
        pairs.append(dict(small_bin_index=i,large_bin_index=j,small_low_um=pair_edges[i],small_high_um=pair_edges[i+1],
            large_low_um=pair_edges[j],large_high_um=pair_edges[j+1],confirmed_collision_lower_bound=int(observed[i,j]),
            predicted_expected_count=float(pooled[i,j]),predicted_poisson_sd=float(np.sqrt(pooled[i,j]))))
    pair_table=pd.DataFrame(pairs);pair_table.to_csv(DATA/'diameter_pair_counts.csv',index=False)
    scenes.append(dict(row_type='total',scene_index=None,raw_scene_num=None,display_label='All scenes',
        confirmed_collision_lower_bound=summary['review']['confirmed'],predicted_expected_count=float(pooled.sum()),exposure_volume_m3_s=EXPOSURE*14))
    scene_table=pd.DataFrame(scenes)
    scene_table['predicted_poisson_sd']=np.sqrt(scene_table.predicted_expected_count)
    scene_table['predicted_density_bootstrap_sd']=0.
    scene_table['predicted_total_sd']=scene_table.predicted_poisson_sd
    for count,rate in [('confirmed_collision_lower_bound','confirmed_collision_rate_lower_bound_m3_s'),('predicted_expected_count','predicted_collision_rate_m3_s'),('predicted_total_sd','predicted_collision_rate_total_sd_m3_s')]:
        scene_table[rate]=scene_table[count]/scene_table.exposure_volume_m3_s
    scene_table.to_csv(DATA/'scene_collision_counts.csv',index=False)
    rows=scene_table.iloc[:-1];total=scene_table.iloc[-1]
    band=pair_table.loc[pair_table.small_low_um.ge(100)&pair_table.small_low_um.lt(150)]
    summary['gravity']={**total.dropna().to_dict(),'observed_to_predicted':float(total.confirmed_collision_lower_bound/total.predicted_expected_count),
        'scenes_below_minus_one_sd':int((rows.confirmed_collision_lower_bound<rows.predicted_expected_count-rows.predicted_poisson_sd).sum()),
        'scenes_above_plus_one_sd':int((rows.confirmed_collision_lower_bound>rows.predicted_expected_count+rows.predicted_poisson_sd).sum()),
        'occupied_pair_bins':int(pair_table.confirmed_collision_lower_bound.gt(0).sum()),
        'largest_observed_pair':pair_table.loc[pair_table.confirmed_collision_lower_bound.idxmax()].to_dict(),
        'small_parent_100_150_observed':int(band.confirmed_collision_lower_bound.sum()),
        'small_parent_100_150_predicted':float(band.predicted_expected_count.sum())}
    densities=pd.DataFrame(densities);densities.to_csv(DATA/'diameter_density_by_scene.csv',index=False)
    frames=pd.concat(frames_all,ignore_index=True);frames.to_parquet(PRIVATE/'frame_metrics.parquet',index=False)
    stats=population_count.scene_summary(pl.from_pandas(frames),sorted(frames.scene.unique()))
    stats.write_csv(DATA/'scene_population_statistics.csv')
    bins,curves=population_size.load_scene_curves(DATA/'diameter_density_by_scene.csv')
    low,high=population_size.bootstrap_mean_interval(curves,replicates=50000,seed=260715,chunk_size=1000)
    bins.assign(mean=curves.mean(axis=0),ci025=low,ci975=high).to_csv(DATA/'diameter_mean_ci.csv',index=False)
    velocity=pd.concat(tracks_all,ignore_index=True);velocity.to_parquet(PRIVATE/'velocity_tracks.parquet',index=False)
    summary['population']={'velocity_tracks':len(velocity),'particle_observations':int(frames.particles_per_hologram.sum()),
        'mean_particles_per_frame':float(frames.particles_per_hologram.mean()),'mean_lwc_g_m3':float(frames.lwc_g_m3.mean()),
        'bootstrap_replicates':50000,'bootstrap_seed':260715,
        'velocity_plot_outside_limits':{c:int((~velocity[c].between(lo,hi)).sum()) for c,lo,hi in [('vx_m_s',-.5,.5),('vy_m_s',-.25,1),('vz_m_s',-.5,.5)]},
        'training_distribution':'The synthetic training population remains frozen; these reprocessed experimental statistics do not regenerate training data.'}
    old_curve=pd.read_csv(REPO/'data/paper_figures/reference/review_topn_yield_precision_260715.csv')
    new_curve=pd.read_csv(DATA/'review_budget.csv')
    comparison=[]
    for n in [10,20,30,40,50,60,70,80]:
        a=old_curve.loc[old_curve.top_n_per_scene.eq(n)].iloc[0];b=new_curve.loc[new_curve.top_n_per_scene.eq(n)].iloc[0]
        comparison.append(dict(reviews_per_scene=n,total_reviews=14*n,public_confirmed=int(a.cumulative_confirmed_collisions),revision_confirmed=int(b.cumulative_confirmed_collisions),
            public_precision=float(a.top_n_precision),revision_precision=float(b.top_n_precision),public_conditional_coverage=float(a.cumulative_confirmed_collisions/185),revision_conditional_coverage=float(b.coverage_of_reviewed_confirmed)))
    pd.DataFrame(comparison).to_csv(DATA/'public_vs_revision_review_budget.csv',index=False)
    summary['created_at']=stamp()
    save(DATA/'results.json',summary)
    print(json.dumps({'review':summary['review'],'gravity':summary['gravity'],'physical':diagnostics},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
