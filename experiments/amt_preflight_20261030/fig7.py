"""Figure 7 diagnostics using the very tracks supplied to collision features."""
from common import *
sys.path.insert(0, str(REPO / 'scripts'))
from figure_support import particle_metrics as metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed = 260745
raw = read_table(tracks_path(seed, 'raw.parquet'))
smooth = read_table(tracks_path(seed, 'geometry_smoothed.parquet'))
# Geometry is the exact CSV-parsed table used for collision image features;
# the boundary probability is retained in the companion tracking table.
smooth = smooth.merge(read_table(tracks_path(seed, 'smoothed.parquet'))[
    ['frame','track_id','postOutProb']], on=['frame','track_id'], validate='one_to_one')
paired = raw[['frame','track_id','x','y']].rename(columns={'x':'raw_x','y':'raw_y'}).merge(
    smooth[['frame','track_id','x','y','z','diameter_px','postOutProb']].rename(columns={
        'x':'smooth_x','y':'smooth_y','z':'smooth_z','postOutProb':'post_out_prob'}),
    on=['frame','track_id'], validate='one_to_one').sort_values(['frame','track_id'],kind='stable').reset_index(drop=True)
original = read_table(REPO / 'data/paper_figures/fig07/paired_tracks_20mm.parquet')
keys = ['frame','raw_x','raw_y']
oldkeys = original[keys].sort_values(keys,kind='stable').reset_index(drop=True)
newkeys = paired[keys].sort_values(keys,kind='stable').reset_index(drop=True)
pd.testing.assert_frame_equal(oldkeys, newkeys, check_dtype=False, check_exact=True)
assert not paired.duplicated(keys).any()
joined = paired.merge(original,on=keys,suffixes=('_shared','_old'),validate='one_to_one')
different = {c: int(np.count_nonzero(joined[c+'_shared'] != joined[c+'_old']))
             for c in ['smooth_x','smooth_y','smooth_z','diameter_px','post_out_prob']}
# Legacy truth's z_slice is a zero-origin physical coordinate divided by dz.
truth_table = read_table(truth_path(seed, 'particles.parquet'))
np.testing.assert_allclose(truth_table.z_slice*100, truth_table.z_um, atol=5.1e-5, rtol=0)
metrics.TRUTH_PATH = truth_path(seed, 'particles.parquet')
truth = metrics.load_truth()
paired['smooth_z'] -= 1
errors = metrics.match_tracks(truth,{c:paired[c].to_numpy() for c in paired})
error_frame = pd.DataFrame({c:errors[c] for c in ['xy_um','z_um','diameter_relative']})
write_table(error_frame, ROOT / 'results/fig7_shared_track_errors.parquet')
pr_path=WORKSPACE/'audit/results/published-figures/figure07/particle_pr_curve.parquet'
summary = {'status':'PASS','seed':seed,'frames':6000,'truth_particle_points':1200000,
           'same_operational_detection_rows':len(paired),'raw_detection_keys_exactly_equal':True,
           'differences_from_old_paired_file':different, 'matched_track_points':len(error_frame),
           'xy_p50_p90_p95_um':np.percentile(errors['xy_um'],[50,90,95]).tolist(),
           'absolute_z_p50_p90_p95_um':np.percentile(np.abs(errors['z_um']),[50,90,95]).tolist(),
           'diameter_p2p5_p5_p50_p95_p97p5_percent':np.percentile(100*errors['diameter_relative'],[2.5,5,50,95,97.5]).tolist(),
           'z_convention':'(one-based reconstructed slice - 1)*100 um; truth already has zero origin',
           'raw_tracks':{'path':str(tracks_path(seed,'raw.parquet')),'sha256':sha256(tracks_path(seed,'raw.parquet'))},
           'smooth_tracks':{'path':str(tracks_path(seed,'geometry_smoothed.parquet')),'sha256':sha256(tracks_path(seed,'geometry_smoothed.parquet'))},
           'boundary_probability_tracks_sha256':sha256(tracks_path(seed,'smoothed.parquet')),
           'number_density_volume_cm3':10.24*10.24*102.4/1000,
           'old_paired_sha256':sha256(REPO/'data/paper_figures/fig07/paired_tracks_20mm.parquet'),
           'truth_sha256':sha256(truth_path(seed,'particles.parquet')),
           'particle_pr_source':{'path':str(pr_path),'sha256':sha256(pr_path),
                                'role':'Previously verified low-confidence detection curve on the same test seed; reused unchanged.'},
           'scope':'Same held-out scene and same corrected measurement tracks as the controlled collision experiments; existing 102.4 mm truth.'}
write_json(summary, ROOT / 'results/fig7_shared_tracks.json')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,3,figsize=(13,7.8),layout='constrained')
pr = read_table(pr_path)
axes[0,0].plot(pr.recall,pr.precision,color='#23689b')
axes[0,0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02),title='(a) Particle detection')
for ax,values,limit,title,xlabel in [
    (axes[0,1],errors['xy_um'],(0,60),'(b) Position in the image plane','Absolute XY error [um]'),
    (axes[0,2],np.abs(errors['z_um'])/1000,(0,5),'(c) Depth in physical coordinates','Absolute Z error [mm]'),
    (axes[1,0],100*errors['diameter_relative'],(-25,25),'(d) Diameter','Relative diameter error [%]')]:
    ax.hist(values,bins=np.linspace(*limit,100),color='#4e8475',alpha=.85)
    if title.startswith('(d)'):
        q=np.percentile(values,[2.5,5,50,95,97.5])
        annotation=f'Median {q[2]:.3g}\nCentral 90% [{q[1]:.3g}, {q[3]:.3g}]\nCentral 95% [{q[0]:.3g}, {q[4]:.3g}]'
    else:
        q = np.percentile(values,[50,90,95])
        annotation='P50 / P90 / P95\n'+ ' / '.join(f'{v:.3g}' for v in q)
    ax.text(.03 if title.startswith('(d)') else .98,.96,annotation,
            ha='left' if title.startswith('(d)') else 'right',va='top',transform=ax.transAxes,fontsize=9)
    ax.set(xlabel=xlabel,ylabel='Matched particle points',xlim=limit,title=title)
inside = truth_table.loc[truth_table.center_in_volume.eq(1)]
edges=np.arange(25,301,25); centers=(edges[:-1]+edges[1:])/2
for data,label,color in [(inside.diameter_um,'Truth','#263c55'),(smooth.diameter_px*10,'Measured','#cb7147')]:
    counts=np.histogram(data,bins=edges)[0]/(6000*summary['number_density_volume_cm3'])
    axes[1,1].plot(centers,counts,'o-',label=label,color=color)
axes[1,1].set(xlabel='Diameter [um]',ylabel='Number density [cm^-3]',title='(e) Same-sequence size distributions')
axes[1,1].legend(frameon=False)
for table,tid,diameter,label,color in [(inside,'particle_id','diameter_um','Truth','#263c55'),
                                      (smooth.assign(diameter_um=smooth.diameter_px*10),'track_id','diameter_um','Measured','#cb7147')]:
    group=table.groupby(tid).agg(length=('frame','size'),diameter=(diameter,'mean'))
    sizes=group.loc[group.diameter.between(25,300,inclusive='left'),'length'].to_numpy()
    lengths=np.arange(1,601)
    axes[1,2].plot(lengths,[(sizes>=n).sum() for n in lengths],label=label,color=color)
axes[1,2].set(yscale='log',xlabel='Minimum trajectory length [frames]',ylabel='Number of trajectories',title='(f) Shared input trajectories')
axes[1,2].legend(frameon=False)
fig.suptitle('Held-out synthetic test: seed 260745 | 6,000 frames | measurement front 74.9 mm | phase retrieval 3 iterations',fontsize=13)
fig.savefig(ROOT/'figures/fig7_shared_test.png',dpi=160)
plt.close(fig)
emit(status='PASS', **{k:summary[k] for k in ['same_operational_detection_rows','differences_from_old_paired_file','matched_track_points','xy_p50_p90_p95_um','absolute_z_p50_p90_p95_um']})
