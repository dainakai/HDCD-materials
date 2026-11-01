"""Recheck manuscript exposure and Poisson uncertainty on the fixed review set."""
from common import *
import shutil
sys.path.insert(0, str(REPO / 'scripts'))
from reproduce_figures import gravity_analysis

_, scenes, pairs, _, expected, sd, _ = gravity_analysis()
old = read_table(WORKSPACE/'audit/results/published-figures/figure14/scene_counts.csv')
previous = read_table(WORKSPACE/'audit/results/manuscript-gravity/figure14/scene_counts.csv')
for col in ['predicted_expected_count','predicted_total_sd','exposure_volume_m3_s']:
    np.testing.assert_allclose(scenes[col],previous[col],atol=1e-12,rtol=0)
np.testing.assert_allclose(sd**2,expected,rtol=1e-14)
assert scenes.predicted_density_bootstrap_sd.eq(0).all()
assert scenes.iloc[-1].confirmed_collision_lower_bound == 185
changes = scenes.loc[scenes.row_type.eq('scene')].copy()
changes['within_1sd'] = abs(changes.confirmed_collision_lower_bound-changes.predicted_expected_count)<=changes.predicted_total_sd
old_within = abs(old.confirmed_collision_lower_bound-old.predicted_expected_count)<=old.predicted_total_sd
changes['historical_within_1sd'] = old_within.iloc[:14].to_numpy()
write_table(scenes,ROOT/'results/R7_scene_counts.csv')
write_table(pairs,ROOT/'results/R7_diameter_pair_counts.csv')
write_table(changes,ROOT/'results/R7_scene_1sd.csv')
total=scenes.iloc[-1]
write_json({'status':'PASS','convention':'manuscript','recorded_frames_per_scene':10917,
            'total_time_s':14*10917/4000,'volume_cm3':10.24*10.24*102.3/1000,
            'total_exposure_m3_s':float(total.exposure_volume_m3_s),
            'observed_events':185,'observed_rate_m3_s':float(total.confirmed_collision_rate_lower_bound_m3_s),
            'predicted_events':float(total.predicted_expected_count),'poisson_sd':float(total.predicted_total_sd),
            'density_bootstrap_sd':0,'scenes_within_1sd':int(changes.within_1sd.sum()),
            'changed_scene_1sd_judgements':int((changes.within_1sd!=changes.historical_within_1sd).sum()),
            'scope':'Historical 185 reviewed experimental events fixed; not a new-model experiment.'},ROOT/'results/R7.json')
for number in [14,15]:
    shutil.copy2(WORKSPACE/f'audit/results/manuscript-gravity/figure{number}.png',ROOT/f'figures/figure{number}_manuscript.png')
emit(status='PASS',expected=float(total.predicted_expected_count),sd=float(total.predicted_total_sd))
