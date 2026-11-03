"""Fresh held-out inference, independent matching checks and explicit ablations."""
from common import *
import argparse
import gc
import torch
from threadpoolctl import threadpool_limits
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from ra3d.models import load_model
from ra3d import minip
from ra3d.cascade import predict_full,ensemble_scores
from ra3d.evaluation import event_curve
from ra3d.config import KEY_COLUMNS,GEOMETRY_COLUMNS
from ra3d._core.context import context_frame
from ra3d._core.relation import relation_scores
from ra3d._core.policy import temporal_mode_predictions
from ra3d._core.residual import correct_predictions

def independent_match(pred,truth):
    rows=[];columns=[];tf=truth.frame.to_numpy();tx=truth.x_um.to_numpy();ty=truth.y_um.to_numpy()
    for i,row in enumerate(pred.itertuples()):
        eligible=np.flatnonzero((np.abs(tf-row.candidate_frame)<=5)&(np.hypot(tx-row.x_um,ty-row.y_um)<=350.))
        rows.extend([i]*len(eligible));columns.extend(eligible.tolist())
    matrix=csr_matrix((np.ones(len(rows),np.int8),(rows,columns)),shape=(len(pred),len(truth)))
    return int((maximum_bipartite_matching(matrix,perm_type='column')>=0).sum())

def evaluate(pred,truth,out,independent=True):
    out.mkdir(parents=True,exist_ok=True)
    if '_source_index' in pred:assert pred._source_index.is_unique
    curve,metrics=event_curve({260745:pred},{260745:truth})
    curve['false_positive_events']=curve.predicted_events-curve.matched_events
    checks=[]
    if independent:
        for target in [.5,.8,.9]:
            candidates=curve.loc[curve.recall>=target]
            if candidates.empty:
                checks.append({'target_recall':target,'status':'UNREACHABLE'});continue
            point=candidates.iloc[0];selected=pred.loc[pred.score>=point.threshold]
            actual=independent_match(selected,truth)
            assert actual==int(point.matched_events)
            checks.append({'target_recall':target,'independent_matched_events':actual,'status':'PASS'})
        actual=independent_match(pred,truth)
        assert actual==int(curve.iloc[-1].matched_events)
        checks.append({'full_predictions':len(pred),'independent_matched_events':actual,'status':'PASS'})
    increments=np.diff(np.r_[0.,curve.recall.to_numpy()])
    envelope=np.maximum.accumulate(curve.precision.to_numpy()[::-1])[::-1]
    assert np.isclose(np.dot(increments,envelope),metrics['event_ap'],atol=1e-12,rtol=0)
    metrics['noninterpolated_recall_weighted_precision']=float(np.dot(increments,curve.precision))
    write_table(pred,out/'predictions.parquet');write_table(curve,out/'precision_recall.csv')
    write_json(metrics,out/'metrics.json');write_json(checks,out/'independent_checks.json')
    return metrics

def stage_prediction(frame,score,bundle,suppression):
    work=frame.copy();work['stage_score']=score
    policy={**bundle['recipe']['source_policy'],'score_column':'stage_score','coordinate_suppression':suppression}
    selected,audit=temporal_mode_predictions(work,seed=260745,method='uncorrected',config=policy)
    pred=correct_predictions(selected,frame,bundle['residual_models'],source_method='uncorrected',method='score')
    return pred,audit

def stages(bundle,truth,static_file=None,full_file=None):
    """Keep the Table 5 common postprocessing explicit, plus an actual removal control."""
    out=ROOT/'results/table5';out.mkdir(exist_ok=True)
    static=read_table(static_file or path(260745,'ours','static_scored.parquet'))
    full=read_table(full_file or path(260745,'ours','paper92.parquet'))
    base=ensemble_scores(bundle['full_models'],full)
    ranks,_=relation_scores(context_frame(full,base),bundle['relation_model'],bundle['relation_mean'],bundle['relation_std'],bundle['recipe'])
    rows=[];audits={}
    for name,frame,score,suppression in [('trajectory',static,static.structural_score,False),
            ('image',full,base,False),('relation',full,ranks,False),('suppression',full,ranks,True)]:
        pred,audit=stage_prediction(frame,score,bundle,suppression)
        metric=evaluate(pred,truth,out/name,independent=False)
        rows.append({'stage':name,**summary_row(metric)})
        audits[name]=audit
    write_table(pd.DataFrame(rows),out/'table5.csv')
    write_json({'common_operations':['temporal modes','maximum 5 modes per terminating track',
        'pre-suppression lineage rank penalty','final 92-feature localization regressors'],
        'localization_feature_availability':'The same trained regressors receive the features available at each stage. Image features are missing in the trajectory-only row. This is not an isolated intervention on classifier scores alone.',
        'candidate_sets':'The trajectory row uses the static shortlist; later rows use the scoring union. The selection effect is included in this stage comparison.',
        'last_row_addition':'coordinate NMS','interpretation':'Table 5 scoring-stage comparison with common downstream processing explicitly stated.',
        'selection':audits},out/'protocol.json')
    # Actual removal of duplicate suppression: all U candidates, no modes,
    # lineage cap, coordinate NMS or lineage penalty. Regression remains fixed
    # so this intervention isolates duplicate suppression from localization.
    candidates=full[KEY_COLUMNS+GEOMETRY_COLUMNS].copy()
    candidates['_source_index']=np.arange(len(full));candidates['uncorrected']=ranks
    candidates['synth_seed']=260745
    pred=correct_predictions(candidates,full,bundle['residual_models'],source_method='uncorrected',method='score')
    metric=evaluate(pred,truth,ROOT/'results/no-duplicate-suppression',independent=False)
    write_json({'removed':['temporal mode representatives','maximum 5 modes','coordinate NMS','lineage rank penalty'],
                'kept':['relation scores','final localization regression'],**summary_row(metric)},ROOT/'results/no-duplicate-suppression/intervention.json')
    return rows

def summary_row(m):
    row={'event_ap':m['event_ap'],'maximum_f1':m['maximum_f1'],'maximum_recall':m['maximum_recall'],
         'prediction_rows':m['predicted_events']}
    for target in [.8,.9]:
        point=m[f'operating_at_recall_{target:.2f}'];suffix=str(target).replace('.','p')
        row[f'precision_at_recall_{suffix}']=None if point is None else point['precision']
        row[f'false_positives_at_recall_{suffix}']=None if point is None else int(point['false_positive_events'])
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours','depth_free','minip2d'],required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--stages',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        truth=events(260745);out=ROOT/f'results/{a.method}'
        full=read_table(path(260745,a.method,'paper92.parquet' if a.method=='ours' else 'full64.parquet'))
        assert not any(c in full for c in ['spatial_positive','identity_training_selected','matched_truth_event_id'])
        if a.method=='ours':
            bundle=load_model(ROOT/'models/ours',device=a.device)
            base=ensemble_scores(bundle['full_models'],full)
            write_table(full[KEY_COLUMNS].assign(base_score=base),out/'candidate_scores.parquet')
            pred,audit=predict_full(full,bundle,scene_id=260745,base_scores=base)
        else:
            bundle=minip.load(ROOT/f'models/{a.method}',device=a.device)
            pred=minip.predict(full,bundle,260745);audit={}
        # Verify manuscript rank convention on surviving representatives.
        assert np.array_equal(pred._lineage_rank,pred._lineage_rank_before_suppression)
        metric=evaluate(pred,truth,out)
        pool=read_table(path(260745,a.method,'shortlist.parquet'))
        pool['score']=1.;_,coverage=event_curve({260745:pool},{260745:truth})
        summary={'method':a.method,**summary_row(metric),'candidate_pool_rows':len(pool),
                 'candidate_pool_recall':coverage['maximum_recall']}
        write_json(summary,out/'summary.json')
        write_json({'status':'PASS','fresh_inference':True,'held_out_seed':260745,
                    'test_truth_used_for_model_fit':False,'test_settings_history':json.loads((ROOT/'config/protocol.json').read_text())['test_usage_note'],
                    'rank_before_suppression_verified':True,
                    'input_sha256':sha256(path(260745,a.method,'paper92.parquet' if a.method=='ours' else 'full64.parquet')),
                    'model_sha256':sha256(ROOT/f'models/{a.method}/model.json'),'policy':audit},out/'verification.json')
        # Prespecified sensitivity only; the production convention remains before.
        if a.method=='ours':
            after_bundle={**bundle,'recipe':{**bundle['recipe'],'source_policy':{**bundle['recipe']['source_policy'],'lineage_rank_order':'after'}}}
            after,_=predict_full(full,after_bundle,scene_id=260745,base_scores=base)
        else:after=minip.predict(full,bundle,260745,lineage_rank_order='after')
        before_geometry=pred.sort_values('_source_index')[['_source_index','candidate_frame','x_um','y_um','z_um']].reset_index(drop=True)
        after_geometry=after.sort_values('_source_index')[before_geometry.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(before_geometry,after_geometry,check_exact=True)
        alternative=evaluate(after,truth,ROOT/f'results/rank_after/{a.method}',independent=False)
        write_json({'production_rank_order':'before','alternative_rank_order':'after','same_candidates_and_coordinates':True,
                    'before':summary_row(metric),'after':summary_row(alternative),
                    'delta_ap_after_minus_before':alternative['event_ap']-metric['event_ap'],
                    'used_to_reselect_final_policy':False},ROOT/f'results/{a.method}/rank_sensitivity.json')
        if a.stages:
            assert a.method=='ours';stages(bundle,truth)
        emit(status='PASS',**summary)
