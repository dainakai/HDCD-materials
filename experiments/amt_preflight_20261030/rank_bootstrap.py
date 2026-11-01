"""Paired temporal-block sensitivity for before/after rank penalties.

Each sampled block is an independent copy of its saved prediction/truth graph.
Integer block multiplicities therefore multiply threshold-wise TP and prediction
increments exactly; no matching has to be recomputed for each bootstrap draw.
This is conditional on the saved model and a single test sequence, not a new-
sequence generalization interval. Five boundary frames are excluded at each end
of every 1,000-frame block to avoid matches across resampled block boundaries.
"""
from common import *
from ra3d.evaluation import event_curve

TRUTH_TEST=read_table(truth_path(260745))
N_REPLICATES=2000


def prepared(pred):
    curves=[]; truths=[]; blocks={}; truth_blocks={}
    for block in range(6):
        low=block*1000+5;high=(block+1)*1000-6
        t=TRUTH_TEST.loc[TRUTH_TEST.frame.between(low,high)].reset_index(drop=True)
        p=pred.loc[pred.candidate_frame.between(low,high)].reset_index(drop=True)
        c,_=event_curve({block:p},{block:t})
        curves.append(c);truths.append(len(t));blocks[block]=p;truth_blocks[block]=t
    all_scores=np.concatenate([c.threshold.to_numpy() for c in curves])
    unique=np.unique(all_scores)[::-1]
    indices=np.searchsorted(-unique,-all_scores)
    increments_tp=np.concatenate([np.diff(np.r_[0,c.matched_events]) for c in curves])
    increments_pred=np.concatenate([np.diff(np.r_[0,c.predicted_events]) for c in curves])
    groups=np.concatenate([np.full(len(c),b) for b,c in enumerate(curves)])
    def ap(weights):
        total=np.dot(weights,truths)
        multiplier=weights[groups]
        tp=np.bincount(indices,weights=increments_tp*multiplier,minlength=len(unique)).cumsum()
        count=np.bincount(indices,weights=increments_pred*multiplier,minlength=len(unique)).cumsum()
        precision=np.divide(tp,count,out=np.ones_like(tp),where=count>0)
        envelope=np.maximum.accumulate(precision[::-1])[::-1]
        return float(np.dot(np.diff(np.r_[0,tp/total]),envelope))
    _,direct=event_curve(blocks,truth_blocks)
    assert np.isclose(ap(np.ones(6)),direct['event_ap'],atol=1e-12,rtol=0)
    return ap,sum(truths)


def run(name,before_path,after_path):
    before=read_table(before_path);after=read_table(after_path)
    f,n=prepared(before);g,n2=prepared(after);assert n==n2
    rng=np.random.default_rng(20261030)
    weights=rng.multinomial(6,np.full(6,1/6),size=N_REPLICATES)
    # The 2,000 draws contain repeated six-block multiplicities. Evaluate
    # each distinct draw once; retain the original draw order and frequency.
    unique,inverse=np.unique(weights,axis=0,return_inverse=True)
    differences=np.array([g(w)-f(w) for w in unique])[inverse]
    row={'arm':name,'block_frames':1000,'guard_frames_per_side':5,'blocks':6,
         'replicates':N_REPLICATES,'truth_after_guards':n,
         'guarded_before_ap':f(np.ones(6)),'guarded_after_ap':g(np.ones(6)),
         'guarded_delta_ap':g(np.ones(6))-f(np.ones(6)),
         'paired_delta_ci025':float(np.quantile(differences,.025)),
         'paired_delta_ci975':float(np.quantile(differences,.975)),
         'bootstrap_fraction_delta_positive':float((differences>0).mean())}
    emit(**row)
    return row


if __name__=='__main__':
    rows=[]
    for path in sorted((ROOT/'results').glob('*/predictions_before.parquet')):
        marker=path.parent/'rank_bootstrap.json'
        if marker.exists():
            rows.append(json.loads(marker.read_text()));continue
        row=run(path.parent.name,path,path.with_name('predictions_after.parquet'))
        write_json(row,marker);rows.append(row)
    write_table(pd.DataFrame(rows),ROOT/'results/R4_paired_block_bootstrap.csv')
