"""Fit every collision stage using the confirmed manuscript interpretation."""
from common import *
import argparse
import time
import torch
from ra3d.training import load_paper_training,fit_staged
from ra3d import minip

def attention_labels(contexts,method):
    # The ranker is supervised on C_v^(20), so its fallback and event weights
    # must be defined on those retained candidates, not on absent U rows.
    audits=[];output=[]
    for context in contexts:
        seed=int(context.synth_seed.iloc[0]);scores=context.score.to_numpy()
        indices=[]
        for local in context.groupby('vanished_track_id',sort=False).indices.values():
            indices.extend(local[np.argsort(-scores[local],kind='stable')[:20]].tolist())
        selected=context.iloc[indices].copy()
        relabeled,audit=labels(selected,seed,method)
        changes=int(np.count_nonzero(selected.identity_training_selected.to_numpy()!=relabeled.identity_training_selected.to_numpy()))
        result=context.copy()
        for col in ['identity_training_selected','relation_positive','spatial_positive','matched_truth_event_id']:
            if col not in result:result[col]=False if col=='relation_positive' else -1
            result.iloc[indices,result.columns.get_loc(col)]=relabeled[col].to_numpy()
        audits.append({'seed':seed,'changed_labels_from_U_scope':changes,**audit});output.append(result)
    write_json({'scope':'Actual top-20 retained attention candidates, separately for each training scene.',
                'same_user_confirmed_identity_rule':True,'scenes':audits},ROOT/f'models/{method}/attention-labels.json')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours','depth_free','minip2d'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();started=time.monotonic()
    if a.device.startswith('cuda'):assert torch.cuda.is_available(),'CUDA required for this invocation'
    output=ROOT/f'models/{a.method}'
    if (output/'model.json').exists():emit(status='ALREADY_COMPLETE',method=a.method);raise SystemExit(0)
    protocol=json.loads((ROOT/'config/protocol.json').read_text())
    assert protocol['rank_order']=='before' and protocol['training_z_gate_um']==15000
    assert protocol['label_rule'].startswith('consistent identity')
    source_manifest={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'src/ra3d').rglob('*.py'))}
    for name in ['train.py','prepare.py','common.py','bootstrap.py','minip_generate.py','measure.py','tracking.py','minip_tracks.py','initialize.py','engine-manifest.json','production-source.json']:
        p=SCRIPT/name
        source_manifest[str(p.relative_to(REPO))]=sha256(p)
    source_manifest['config/protocol.json']=sha256(ROOT/'config/protocol.json')
    for s in TRAIN:
        assert path(s,a.method,'images.json').exists()
    if a.method=='ours':
        stages,manifest=load_paper_training(ROOT/'data')
        manifest.update(source_manifest)
        bundle,audit=fit_staged(stages,output,device=a.device,cpu_threads=a.threads,
              source_manifest=manifest,verify_paper=False,progress=lambda v:emit(method=a.method,**v),
              relation_context_transform=lambda contexts:attention_labels(contexts,a.method))
    else:
        bundle,audit=minip.train(ROOT/'data',output,'depth-free' if a.method=='depth_free' else 'minip-2d',
              device=a.device,cpu_threads=a.threads,progress=lambda v:emit(method=a.method,**v),
              relation_context_transform=lambda contexts:attention_labels(contexts,a.method))
    write_json({'status':'PASS','method':a.method,'protocol_sha256':sha256(ROOT/'config/protocol.json'),
                'source_files':source_manifest,'training_seeds':TRAIN,'test_seed_used_in_fit':False,
                'upstream_selection_recomputed':True,'base_oof_scores_recomputed':True,
                'reused_pretrained_collision_weights':False,'seconds':time.monotonic()-started,
                'torch':torch.__version__,'device':a.device},output/'complete.json')
    emit(status='PASS',method=a.method,seconds=time.monotonic()-started)
