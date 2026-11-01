"""Independent checks of interventions, test split, AP and event associations."""
from common import *
from ra3d.config import RAW_IMAGE_FEATURES, IMAGE_FEATURES
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

expected=['strict_phase12_z15','legacy_phase12_z15','strict_phase3_z15','legacy_phase3_z15',
          'strict_phase3_z5','strict_phase3_z30','strict_phase3_znone']
truth=read_table(truth_path(260745))
tf=truth.frame.to_numpy();tx=truth.x_um.to_numpy();ty=truth.y_um.to_numpy()


def independent_matching(pred):
    rr=[];cc=[]
    for i,p in enumerate(pred.itertuples()):
        js=np.flatnonzero((abs(tf-p.candidate_frame)<=5)&(np.hypot(tx-p.x_um,ty-p.y_um)<=350))
        rr.extend([i]*len(js));cc.extend(js.tolist())
    matrix=csr_matrix((np.ones(len(rr),np.int8),(rr,cc)),shape=(len(pred),len(truth)))
    return int((maximum_bipartite_matching(matrix,perm_type='column')>=0).sum())


checks=[]
for seed in SEEDS:
    marker=json.loads(data_path(seed,'images3.json').read_text())
    assert marker['status']=='PASS' and marker['acquisition_selection_and_rendered_frames_identical']
    assert marker['output_sha256']==sha256(data_path(seed,'paper92_phase3.parquet'))
    assert (marker['iterations'],marker['padding_px'],marker['front_um'])==(3,2048,74900)
    differences=read_table(data_path(seed,'feature_differences.csv')).set_index('feature')
    for feature in IMAGE_FEATURES:
        expected_valid=marker['image_candidates'] if feature in RAW_IMAGE_FEATURES else marker['full_candidates']
        assert differences.loc[feature,'valid']==expected_valid
        assert differences.loc[feature,'missing_phase12']==differences.loc[feature,'missing_phase3']
    checks.append({'kind':'phase_input','seed':seed,'status':'PASS'})
for name in expected:
    complete=json.loads((ROOT/f'models/{name}/complete.json').read_text())
    assert complete['train_seeds']==TRAIN and complete['test_used_in_fitting'] is False
    training=json.loads((ROOT/f'models/{name}/training_audit.json').read_text())
    assert training['training_scenes']==TRAIN
    assert not any('seed260745' in path for path in training['source_manifest'])
    label_audits=json.loads((ROOT/f'models/{name}/labels.json').read_text())
    assert {item['seed'] for item in label_audits}==set(TRAIN)
    paths=ROOT/'results'/name
    original=read_table(paths/'predictions_before.parquet')
    revised=read_table(paths/'predictions_after.parquet')
    columns=['_source_index','candidate_frame','x_um','y_um','z_um']
    pd.testing.assert_frame_equal(original[columns].sort_values('_source_index').reset_index(drop=True),
                                 revised[columns].sort_values('_source_index').reset_index(drop=True),check_exact=True)
    for rank,pred in [('before',original),('after',revised)]:
        curve=read_table(paths/f'pr_{rank}.csv')
        metrics=json.loads((paths/f'metrics_{rank}.json').read_text())
        ap=np.dot(np.diff(np.r_[0.,curve.recall]),np.maximum.accumulate(curve.precision.to_numpy()[::-1])[::-1])
        np.testing.assert_allclose(ap,metrics['event_ap'],atol=1e-12,rtol=0)
        for target in [.8,.9]:
            point=metrics[f'operating_at_recall_{target:.2f}']
            if point is None:continue
            selected=pred.loc[pred.score>=point['threshold']]
            matched=independent_matching(selected)
            assert matched==int(point['matched_events'])
            checks.append({'kind':'independent_scipy_matching','arm':name,'rank':rank,'target':target,
                           'matched':matched,'predictions':len(selected),'status':'PASS'})
    emit(stage='verified',arm=name)
write_json({'status':'PASS','arms':expected,'checks':checks,
            'scope':'Controlled experiments only; upstream fixed and original holograms retained.'},ROOT/'results/verification.json')
