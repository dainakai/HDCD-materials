"""Additional audit tables; no relabeling or post-hoc exclusion from primary figures."""
import sys
import numpy as np
import pandas as pd
from common import *
from analyze import metrics
sys.path.insert(0,str(REPO/'scripts'))
from reproduce_figures import gravity_analysis

def main():
    data=read(DATA/'results.json')
    new=pd.read_parquet(PRIVATE/'physical_events_all.parquet')
    old=pd.read_csv(REPO/'data/paper_figures/reference/physical_consistency_events.csv')
    rows=[]
    for version,frame in [('public',old),('revision',new)]:
        for role in ['TP','FP']:
            base=frame.loc[frame.classification_role.eq(role)&frame.velocity_available]
            for n in [3,5,10]:
                chosen=base.loc[base[['pre_vy_vanished_n','pre_vy_survivor_n','post_vy_n']].ge(n).all(axis=1)]
                rows.append(dict(version=version,role=role,min_points_per_window=n,
                    **metrics(chosen,'expected_post_vy_m_s','measured_post_vy_m_s')))
    pd.DataFrame(rows).to_csv(DATA/'velocity_window_sensitivity.csv',index=False)
    tp=new.loc[new.status.eq('collision')].copy()
    tp['velocity_residual_m_s']=tp.measured_post_vy_m_s-tp.expected_post_vy_m_s
    tp['diameter_residual_um']=tp.measured_post_diameter_um-tp.expected_post_diameter_um
    top=tp.loc[tp.velocity_available].sort_values('velocity_residual_m_s',key=abs,ascending=False).head(8)
    fields=['scene_num','review_rank','candidate_frame_raw','expected_post_vy_m_s','measured_post_vy_m_s','velocity_residual_m_s',
        'pre_vy_vanished_n','pre_vy_survivor_n','post_vy_n','expected_post_diameter_um','measured_post_diameter_um','z_um_raw']
    top[fields].to_csv(DATA/'velocity_largest_residuals.csv',index=False)
    missing=tp.loc[~tp.diameter_available|~tp.velocity_available]
    missing[['scene_num','review_rank','diameter_available','velocity_available','pre_diameter_vanished_n','pre_diameter_survivor_n','post_diameter_n',
        'pre_vy_vanished_n','pre_vy_survivor_n','post_vy_n']].to_csv(DATA/'confirmed_measurement_availability.csv',index=False)
    data['diagnostics']={'largest_velocity_squared_error_share':float(top.iloc[0].velocity_residual_m_s**2/(tp.velocity_residual_m_s**2).sum()),
        'top_two_velocity_squared_error_share':float((top.iloc[:2].velocity_residual_m_s**2).sum()/(tp.velocity_residual_m_s**2).sum()),
        'confirmed_duplicate_ordered_track_pairs':int(tp.duplicated(['scene_num','vanished_track_id','survivor_track_id']).sum()),
        'primary_exclusion_policy':'No outlier removal. Diameter requires both parent measurements and a post diameter. Velocity requires all three windows with >=3 points, as in the public code.',
        'pair_map_exception':'S0015 rank 34 lacks one parent diameter; remains in 186 total events, excluded from the 185 diameter-pair assignments.'}
    hist=[]
    for status in ['collision','non_collision']:
        bins=np.linspace(.98,1.,41);h=np.histogram(new.loc[new.status.eq(status),'score'],bins)[0]
        assert h.sum()==new.status.eq(status).sum()
        for lo,hi,count in zip(bins[:-1],bins[1:],h):hist.append(dict(status=status,score_low=lo,score_high=hi,count=int(count)))
    pd.DataFrame(hist).to_csv(DATA/'score_histogram.csv',index=False)
    _,old_common,_,*_=gravity_analysis('manuscript')
    old_common.to_csv(DATA/'public_with_revision_gravity_convention.csv',index=False)
    old_saved=pd.read_csv(REPO/'data/paper_figures/reference/scene_collision_counts_observed_vs_hall_260715.csv').iloc[-1]
    new_total=pd.read_csv(DATA/'scene_collision_counts.csv').iloc[-1]
    comparisons=[]
    for name,row in [('public_saved_figure',old_saved),('public_measurements_common_convention',old_common.iloc[-1]),('revision',new_total)]:
        comparisons.append(dict(version=name,confirmed=int(row.confirmed_collision_lower_bound),expected=float(row.predicted_expected_count),
            standard_deviation=float(row.predicted_total_sd),exposure_m3_s=float(row.exposure_volume_m3_s),
            observed_rate_m3_s=float(row.confirmed_collision_rate_lower_bound_m3_s),expected_rate_m3_s=float(row.predicted_collision_rate_m3_s),
            observed_to_predicted=float(row.confirmed_collision_lower_bound/row.predicted_expected_count)))
    pd.DataFrame(comparisons).to_csv(DATA/'public_vs_revision_gravity.csv',index=False)
    data['gravity_comparison']=comparisons
    data['interpretation']={'unreviewed_events':'The review covers exactly the selected top80 per scene; absolute recall remains unknown.',
        'comparison':'Public and revised cohorts were separately ranked and reviewed. Aggregate differences combine pipeline, selected candidates, numerical hardware and review differences; they do not isolate a single causal change.',
        'training':'Frozen synthetic models were not retrained on experimental judgments. Recomputed population plots do not change the original empirical distribution used to generate training data.'}
    save(DATA/'results.json',data)
    print('Added sensitivity, availability, histogram and common-convention baseline tables.')

if __name__=='__main__':main()
