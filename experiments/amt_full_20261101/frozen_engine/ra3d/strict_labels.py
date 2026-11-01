"""Manuscript identity-first labels, with user-fixed t-1/t+1 observations.

Eligibility uses time, XY, and the configured optional Z gate. All three particle roles must be observed.
Candidates near an event with an ID-consistent explanation are neutral unless
they satisfy that identity; events without one use spatial fallback.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._core.matching import candidate_truth_adjacency, incremental_maximum_matching


def attach_identity_labels(candidates, truth, mapping, *, z_tolerance_um=None):
    out=candidates.reset_index(drop=True).copy()
    true=truth.reset_index(drop=True).copy()
    projected=out[['candidate_frame','x_um','y_um','z_um']].copy()
    # Fresh candidate tables already use physical z=(slice-1)*100 um.
    target=true.copy()
    if z_tolerance_um is None:
        projected['z_um']=0.;target['z_um']=0.
    adjacency=candidate_truth_adjacency(projected,target,z_tolerance_um=(15000. if z_tolerance_um is None else z_tolerance_um))
    _,matched=incremental_maximum_matching(range(len(out)),adjacency)
    lookup=mapping.set_index(['frame','observed_track_id']).source_particle_id
    assert lookup.index.is_unique
    times=out.candidate_frame.to_numpy(np.int64)
    def at(track,offset):
        index=pd.MultiIndex.from_arrays([times+offset,out[track].to_numpy(np.int64)])
        return lookup.reindex(index).fillna(-1).to_numpy(np.int64)
    vanished=at('vanished_track_id',-1)
    parent=at('survivor_track_id',-1)
    daughter=at('survivor_track_id',1)
    assigned=np.full(len(out),-1,dtype=np.int32)
    strict=np.zeros(len(out),dtype=bool)
    state=np.full(len(out),'negative',dtype=object)
    roles=true[['parent1_id','parent2_id','child_id']].to_numpy(np.int64)
    for i,eligible in adjacency.items():
        assigned[i]=eligible[0]
        if min(vanished[i],parent[i],daughter[i])<0:
            state[i]='identity_unknown'
            continue
        state[i]='identity_inconsistent'
        for j in eligible:
            p1,p2,ch=roles[j]
            if daughter[i]==ch and ((vanished[i]==p1 and parent[i]==p2) or (vanished[i]==p2 and parent[i]==p1)):
                assigned[i]=j;strict[i]=True;state[i]='identity_consistent';break
    strict_events=set(assigned[strict].tolist())
    selected=strict.copy()
    for i,eligible in adjacency.items():
        if strict[i]:continue
        # Prefer any still-uncovered event over a spatially closer event whose
        # identity-consistent explanation already exists.
        fallback=[j for j in eligible if j not in strict_events]
        if fallback:
            assigned[i]=fallback[0];selected[i]=True
    spatial=assigned>=0
    out['spatial_positive']=spatial.astype(np.int8)
    out['assigned_truth_index']=assigned
    out['identity_state']=state
    out['identity_training_selected']=selected
    out['relation_positive']=selected
    out['vanished_source_particle_id']=vanished
    out['survivor_source_particle_id']=parent
    out['daughter_source_particle_id']=daughter
    out['matched_truth_event_id']=-1
    out.loc[spatial,'matched_truth_event_id']=true.iloc[assigned[spatial]].collision_event_id.to_numpy(np.int64)
    out['truth_sampling_component']=''
    if 'sampling_component' in true:
        out.loc[spatial,'truth_sampling_component']=true.iloc[assigned[spatial]].sampling_component.to_numpy()
    for mask,name in [(spatial,'spatial_event_factor'),(selected,'identity_event_factor')]:
        out[name]=0.
        if mask.any():
            counts=out.loc[mask].groupby('matched_truth_event_id').matched_truth_event_id.transform('size')
            out.loc[mask,name]=(1./counts).to_numpy()
    covered=set(assigned[spatial].tolist())
    audit={'candidate_rows':len(out),'spatial_positive_rows':int(spatial.sum()),
           'spatial_negative_rows':int((~spatial).sum()),'exact_covered_events':len(matched),
           'attributed_events':len(covered),'identity_consistent_events':len(strict_events),
           'identity_fallback_events':len(covered-strict_events),
           'identity_training_positive_rows':int(selected.sum()),
           'neutral_spatial_rows':int((spatial & ~selected).sum()),
           'identity_window':'strict t-1 parents and t+1 daughter; no imputation',
           'gate':{'frames':5,'xy_um':350,'z_um':z_tolerance_um,'candidate_z_origin_correction_um':0},
           'identity_state_rows':{str(k):int(v) for k,v in pd.Series(state).value_counts().items()}}
    components=[]
    for label,group in true.groupby('sampling_component') if 'sampling_component' in true else [('all',true)]:
        indices=set(group.index)
        components.append({'sampling_component':label,'truth_events':len(indices),
                           'exact_covered_events':len(indices & set(matched)),
                           'attributed_events':len(indices & covered),
                           'identity_consistent_events':len(indices & strict_events)})
    return out,audit,pd.DataFrame(components)


def relation_weights(contexts,labels,mask,sources):
    """Equal event mass across scenes; equal positive/negative total mass.

Neutral spatial candidates remain attention inputs, with zero loss weight.
"""
    weights=np.zeros(labels.shape,dtype=np.float32)
    positives=[];negatives=[];event_keys=[];audits=[]
    for context_id,context in enumerate(contexts):
        pos=np.argwhere(mask & (sources[:,:,0]==context_id))
        indices=sources[pos[:,0],pos[:,1],1]
        chosen=context.iloc[indices].identity_training_selected.to_numpy(bool)
        spatial=context.iloc[indices].spatial_positive.to_numpy(bool)
        assert np.array_equal(labels[pos[:,0],pos[:,1]],chosen)
        positives.extend(pos[chosen].tolist())
        negatives.extend(pos[~spatial].tolist())
        event_keys.extend((context_id,int(x)) for x in context.iloc[indices[chosen]].matched_truth_event_id)
        audits.append({'context':context_id,'positive':int(chosen.sum()),'negative':int((~spatial).sum()),
                       'neutral':int((spatial & ~chosen).sum())})
    if not positives or not negatives:raise ValueError('Both supervised classes are required')
    from collections import Counter
    counts=Counter(event_keys)
    factors=np.array([1./counts[k] for k in event_keys]);factors/=factors.sum()
    pp=np.asarray(positives);nn=np.asarray(negatives)
    weights[pp[:,0],pp[:,1]]=factors
    weights[nn[:,0],nn[:,1]]=1./len(nn)
    weights/=weights[mask].mean()
    return weights,audits
