"""Validate the completed campaign and prepare a reviewable material snapshot."""
from common import *
import shutil
import tarfile
import time


def main():
    assert json.loads((ROOT/'campaign-state.json').read_text())['status']=='PASS'
    required=[];summaries={};verification={}
    for seed in SEEDS:
        for relative in [f'runs/render/seed{seed}/complete.json',f'runs/minip/seed{seed}/complete.json',
                         f'runs/measurement/seed{seed}/complete.json',f'runs/tracking/seed{seed}/complete.json',
                         f'runs/minip-tracks/seed{seed}/complete.json']:
            p=ROOT/relative;record=json.loads(p.read_text());assert record['status']=='PASS';required.append(p)
        for method in ['ours','depth_free','minip2d']:
            p=path(seed,method,'images.json');assert json.loads(p.read_text())['status']=='PASS';required.append(p)
    for method in ['ours','depth_free','minip2d']:
        out=ROOT/f'results/{method}';summaries[method]=json.loads((out/'summary.json').read_text())
        verification[method]=json.loads((out/'verification.json').read_text())
        check=verification[method];assert check['status']=='PASS' and check['rank_before_suppression_verified']
        assert check['model_sha256']==sha256(ROOT/f'models/{method}/model.json')
        assert check['input_sha256']==sha256(path(260745,method,'paper92.parquet' if method=='ours' else 'full64.parquet'))
        independent=json.loads((out/'independent_checks.json').read_text())
        assert all(r['status'] in ['PASS','UNREACHABLE'] for r in independent)
        assert json.loads((ROOT/f'models/{method}/complete.json').read_text())['training_seeds']==TRAIN
    manuscript=ROOT/'manuscript';manuscript.mkdir(exist_ok=True)
    metrics=[('Average precision','event_ap'),('Maximum $F_1$','maximum_f1'),
             ('Precision at recall $=0.8$','precision_at_recall_0p8'),('Precision at recall $=0.9$','precision_at_recall_0p9'),
             ('Candidate-pool recall','candidate_pool_recall')]
    lines=[r'\begin{tabular}{lrrr}',r'\tophline',r'Metric & Ours & Depth-free variant & MinIP-based 2D \\',r'\middlehline']
    for title,key in metrics:
        vals=[summaries[m][key] for m in ['ours','depth_free','minip2d']]
        cells=[]
        for i,value in enumerate(vals):
            cells.append('--' if value is None else f'{value:.3f}'+('' if i==0 or vals[0] is None else f' (${{{value-vals[0]:+.3f}}}$)'))
        lines.append(title+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomhline',r'\end{tabular}']);(manuscript/'table3.tex').write_text('\n'.join(lines)+'\n')
    stages=read_table(ROOT/'results/table4/table4.csv')
    lines=[r'\begin{tabular}{lrrrrr}',r'\tophline',r'Stage & AP & P80 & FP80 & P90 & FP90 \\',r'\middlehline']
    for row in stages.to_dict('records'):
        cells=[row['stage']]
        for key in ['event_ap','precision_at_recall_0p8','false_positives_at_recall_0p8','precision_at_recall_0p9','false_positives_at_recall_0p9']:
            value=row[key];cells.append('--' if pd.isna(value) else (f'{int(value):,}' if key.startswith('false_') else f'{value:.3f}'))
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomhline',r'\end{tabular}']);(manuscript/'table4.tex').write_text('\n'.join(lines)+'\n')
    claims={}
    for key in ['event_ap','maximum_f1','precision_at_recall_0p8','precision_at_recall_0p9']:
        values=[summaries[m][key] for m in ['ours','depth_free','minip2d']]
        claims['ours_exceeds_both_'+key]=None if any(x is None for x in values) else values[0]>max(values[1:])
    for key in ['event_ap','precision_at_recall_0p8','precision_at_recall_0p9']:
        values=stages[key].to_numpy();claims['stagewise_nondecreasing_'+key]=bool(np.isfinite(values).all() and (np.diff(values)>=0).all())
    for key in ['false_positives_at_recall_0p8','false_positives_at_recall_0p9']:
        values=stages[key].to_numpy();claims['stagewise_nonincreasing_'+key]=bool(np.isfinite(values).all() and (np.diff(values)<=0).all())
    reductions={}
    for key in ['false_positives_at_recall_0p8','false_positives_at_recall_0p9']:
        for a,b in [('minip2d','depth_free'),('minip2d','ours'),('depth_free','ours')]:
            old,new=summaries[a][key],summaries[b][key]
            reductions[f'{a}_to_{b}_{key}_percent']=None if old in [None,0] or new is None else 100*(old-new)/old
    write_json({'methods':summaries,'table4':stages.to_dict('records'),'false_positive_reductions':reductions},manuscript/'numeric-replacements.json')
    write_json({'claims':claims,'interpretation':'False claims require rewriting the narrative; do not preserve the old conclusions automatically.'},manuscript/'claim-checks.json')
    export=ROOT/'artifacts';export.mkdir(exist_ok=True)
    archive=export/'hdcd-amt-102p3mm-phase3-models.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for method in summaries:
            for name in ['model.json','trees.skops','relation.safetensors','normalization.npz','training_audit.json','complete.json']:
                p=ROOT/f'models/{method}/{name}';tar.add(p,arcname=f'models/{method}/{name}',recursive=False)
        for p in sorted((SCRIPT/'frozen_engine').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='src/'+str(p.relative_to(SCRIPT/'frozen_engine')),recursive=False)
        for name in ['protocol.json','engine-manifest.json','README.md','environment.json']:
            tar.add(SCRIPT/name,arcname=name,recursive=False)
    results=SCRIPT/'results';results.mkdir(exist_ok=True)
    selected=[ROOT/'results/table3.csv',ROOT/'results/scenario-comparison.csv',ROOT/'results/table4/table4.csv',ROOT/'results/table4/protocol.json',
              ROOT/'results/no-duplicate-suppression/intervention.json',ROOT/'results/figure07/summary.json',ROOT/'results/figure9.json',
              manuscript/'numeric-replacements.json',manuscript/'claim-checks.json',manuscript/'table3.tex',manuscript/'table4.tex']
    for method in summaries:
        selected.extend(ROOT/f'results/{method}/{name}' for name in ['metrics.json','summary.json','verification.json','independent_checks.json','rank_sensitivity.json'])
    for name in ['figure07','figure08','figure09','comparison']:
        selected.extend(ROOT/f'figures/{name}.{suffix}' for suffix in ['png','pdf'])
    manifest={}
    for p in selected:
        relative=p.relative_to(ROOT);target=results/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        manifest[str(relative)]=sha256(target)
    profile={'version':'amt-102p3mm-phase3','protocol_sha256':sha256(ROOT/'config/protocol.json'),
             'models':{m:{'directory':str(ROOT/f'models/{m}'),'model_json_sha256':sha256(ROOT/f'models/{m}/model.json')} for m in summaries},
             'review_labels':'Start empty for new candidates; old reviews are comparison evidence only.',
             'raw_image_status':'Original experimental image drive remains unavailable; per-scene calibration must be rechecked before processing.',
             'experimental_reprocessing_performed':False,'gravity_reference_convention':'manuscript'}
    preparation=ROOT.parent/'revision-review-20261101/experimental-preparation'
    write_json(profile,preparation/'model-freeze.json')
    final={'status':'PASS','numerical_outputs_complete':True,'numerical_outputs_manifest':manifest,
           'manuscript_numeric_only_ready':False,'remaining_text_changes':str(SCRIPT/'manuscript-changes.json'),
           'source_test_usage_note':json.loads((ROOT/'config/protocol.json').read_text())['test_usage_note'],
           'experimental_model_frozen':True,'experimental_raw_inputs_complete':False,
           'model_archive':str(archive),'model_archive_sha256':sha256(archive),'completed_at':dt.datetime.now().astimezone().isoformat()}
    write_json(final,ROOT/'finalized.json');write_json(final,results/'manifest.json');emit(stage='finalized',status='PASS')


if __name__=='__main__':main()
