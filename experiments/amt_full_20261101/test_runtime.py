"""Scientific regressions specific to new, physical-origin candidate tables."""
import numpy as np
import pandas as pd
from ra3d.strict_labels import attach_identity_labels
from ra3d._core.policy import temporal_mode_predictions
from ra3d._core.two_d_relation import policy_predictions
from ra3d._core.candidates import vector_um
from ra3d.config import Units


def test_new_candidates_do_not_receive_old_bank_z_correction():
    candidates=pd.DataFrame({'candidate_frame':[10,10],'vanished_track_id':[1,1],
                            'survivor_track_id':[2,2],'x_um':[0.,0.],'y_um':[0.,0.],'z_um':[0.,80.]})
    truth=pd.DataFrame({'frame':[10],'x_um':[0.],'y_um':[0.],'z_um':[0.],
                        'parent1_id':[101],'parent2_id':[102],'child_id':[103],'collision_event_id':[1]})
    mapping=pd.DataFrame({'frame':[9,9,11],'observed_track_id':[1,2,2],
                          'source_particle_id':[101,102,103]})
    result,audit,_=attach_identity_labels(candidates,truth,mapping,z_tolerance_um=50.)
    assert result.identity_training_selected.tolist()==[True,False]
    assert audit['gate']['candidate_z_origin_correction_um']==0
    projected,_,_=attach_identity_labels(candidates,truth,mapping,z_tolerance_um=None)
    assert projected.identity_training_selected.tolist()==[True,True]


def test_both_method_families_preserve_pre_suppression_rank():
    frame=pd.DataFrame({'vanished_track_id':[2,1,1],'survivor_track_id':[20,10,11],
                         'candidate_frame':[100,100,200],'vanished_end_frame':[100,100,200],
                         'time_offset_frame':[0,0,0],'x_um':[0.,10.,10000.],
                         'y_um':[0.,0.,0.],'z_um':[0.,0.,0.],'base_percentile':[.9,.8,.7]})
    config={'pair_mode_gap_frames':10,'maximum_modes_per_vanished_track':5,
            'coordinate_suppression':True,'lineage_log_odds_penalty':1.,'lineage_rank_order':'before'}
    three,_=temporal_mode_predictions(frame,seed=1,method='score',config=config)
    two,_=policy_predictions(frame,frame.base_percentile.to_numpy(),1)
    assert three._source_index.tolist()==two._source_index.tolist()==[0,2]
    assert three._lineage_rank.tolist()==two._lineage_rank.tolist()==[0,1]
    np.testing.assert_array_equal(three.score,two.relation_attention_2d_uncorrected)
    np.testing.assert_array_equal(two._lineage_rank,two._lineage_rank_before_suppression)
    after,_=policy_predictions(frame,frame.base_percentile.to_numpy(),1,lineage_rank_order="after")
    assert after._lineage_rank.tolist()==[0,0]
    np.testing.assert_array_equal(after._lineage_rank_before_suppression,[0,1])


def test_candidate_planes_cover_exact_physical_volume():
    params=Units().candidate_parameters()
    assert vector_um(0.,0.,1.,params)[2]==0.
    assert vector_um(0.,0.,1024.,params)[2]==102300.
