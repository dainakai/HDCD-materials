"""Fresh particle PR and all Figure 7 panels from the collision test input manifest."""
from common import *
import importlib.util
import math
import os
import subprocess
import sys
sys.path.append(str(REPO/'scripts'))
from figure_support import particle_metrics as metrics, particle_plot as plot, particle_density as density, trajectory_length as length
from figure_support.fonts import register_fonts
from figure_support.delivery import render


def main():
    seed=260745;folder=ROOT/'results/figure07';folder.mkdir(exist_ok=True)
    source=REPO/'reference/measurement'
    low=folder/'low_confidence_detections.parquet'
    if not low.exists():
        command=[str(GPU_PYTHON),str(source/'src/detection/raw_detect_best.py'),
                 '--image-dir',str(minip_dir(seed)),'--weights',str(source/'models/yolo/best.pt'),
                 '--out-csv',str(folder/'low_confidence_detections.csv'),
                 '--frame-stats-csv',str(folder/'frame_stats.csv'),'--summary-json',str(folder/'detection.json'),
                 '--conf','0.001','--iou','0.15','--max-det','600','--device','cuda:0',
                 '--batch-size','8','--image-load-workers','4']
        subprocess.run(command,cwd=source,check=True,env=os.environ.copy())
        frame=pd.read_csv(folder/'low_confidence_detections.csv')
        assert frame.frame.nunique()==6000
        write_table(frame,low)
    truthfile=truth_path(seed,'particles.parquet')
    metrics.TRUTH_PATH=truthfile;metrics.LOW_CONF_DETECTIONS=low
    truth=metrics.load_truth();detections,flags=metrics.greedy_detection_flags(truth)
    curve=metrics.precision_recall(detections.conf.to_numpy(),flags,1200000)
    write_table(pd.DataFrame({k:curve[k] for k in ['confidence','recall','precision','f1']}),folder/'particle_pr_curve.parquet')
    actual=measurement(seed)/'raw_detections.csv'
    operational=folder/'operational_detections.parquet';write_table(read_table(actual),operational)
    metrics.LOW_CONF_DETECTIONS=operational;op,opflags=metrics.greedy_detection_flags(truth)
    tp=int(opflags.sum());point={'confidence':.1,'predictions':len(op),'true_positives':tp,
                               'precision':tp/len(op),'recall':tp/1200000}
    track=tracking(seed,'ours');raw=read_table(track/'raw.parquet');smooth=read_table(track/'smoothed.parquet')
    assert raw[['frame','track_id']].equals(smooth[['frame','track_id']])
    paired=raw[['frame','track_id','x','y']].rename(columns={'x':'raw_x','y':'raw_y'}).merge(
        smooth[['frame','track_id','x','y','z','diameter_px','postOutProb']].rename(columns={
            'x':'smooth_x','y':'smooth_y','z':'smooth_z','postOutProb':'post_out_prob'}),
        on=['frame','track_id'],validate='one_to_one').sort_values(['frame','track_id'],kind='stable').reset_index(drop=True)
    paired['smooth_z']-=1.
    errors=metrics.match_tracks(truth,{c:paired[c].to_numpy() for c in paired})
    write_table(pd.DataFrame({k:errors[k] for k in ['xy_um','z_um','diameter_relative']}),folder/'matched_particle_errors.parquet')
    inside=read_table(truthfile);inside=inside.loc[inside.center_in_volume.eq(1)]
    volume=10.24*10.24*102.3/1000.;edges=np.arange(25.,301.,5.)
    bins=pd.DataFrame({'diameter_bin_low_um':edges[:-1],'diameter_bin_high_um':edges[1:]})
    for name,diameters in [('truth',inside.diameter_um),('measured',smooth.diameter_px*10.)]:
        bins[name+'_number_density_cm3_per_5um_bin']=np.histogram(diameters,bins=edges)[0]/(6000*volume)
    write_table(bins,folder/'number_density.csv')
    for name,table,idcol in [('truth',inside,'particle_id'),('measured',smooth.assign(diameter_um=smooth.diameter_px*10.),'track_id')]:
        grouped=table.groupby(idcol).agg(length=('frame','size'),diameter_um=('diameter_um','median')).reset_index()
        write_table(grouped,folder/f'{name}_trajectory_summary.csv')
    summary={'status':'PASS','test_seed':seed,'frames':6000,'truth_points':1200000,
             'source_is_collision_test_tracks':True,'particle_ap':curve['average_precision'],
             'tracking_input_operating_point':point,'measured_points':len(smooth),
             'matched_points':int(errors['matched_count'][0]),
             'xy_p50_p90_p95_um':np.percentile(errors['xy_um'],[50,90,95]).tolist(),
             'absolute_z_p50_p90_p95_um':np.percentile(abs(errors['z_um']),[50,90,95]).tolist(),
             'diameter_p2p5_p5_p50_p95_p97p5_percent':np.percentile(100*errors['diameter_relative'],[2.5,5,50,95,97.5]).tolist(),
             'number_density_volume_cm3':volume,'physical_z':'(one-based slice - 1)*100 um',
             'particle_pr_rule':'Confidence-ranked nearest-unmatched truth in an 8 px radial gate; same historical rule.',
             'position_error_rule':'Raw XY squared-distance Hungarian association in an 8 px gate; errors from the exact collision-feature smoothed tracks.',
             'inputs':{str(p.relative_to(ROOT)):sha256(p) for p in [truthfile,low,actual,track/'raw.parquet',track/'smoothed.parquet',minip_dir(seed)/'identity.json']}}
    write_json(summary,folder/'summary.json')
    from figure_support.synthetic_particles import main as draw
    draw(folder,folder/'panels',ROOT/'figures')
    emit(stage='figure7_complete',**summary)

if __name__=='__main__':main()
