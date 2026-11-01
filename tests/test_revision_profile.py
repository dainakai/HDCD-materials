"""Scientific invariants for the proposed depth and rank-order interventions."""
import numpy as np
import pandas as pd
import pytest
from ra3d.simulation.physical import SynthParams
from ra3d._core.policy import temporal_mode_predictions


def test_explicit_volume_span_covers_1024_reconstruction_centers():
    legacy = SynthParams()
    revised = SynthParams(volume_depth_um=102300)
    grid = np.arange(revised.slices)*revised.dz_um
    assert legacy.depth_um == 102400  # Original generation remains reproducible.
    assert len(grid) == 1024 and grid[0] == 0 and grid[-1] == revised.depth_um
    assert revised.c002_front_distance_um + grid[-1] == 177200
    assert revised.width_um == legacy.width_um and revised.height_um == legacy.height_um
    for depth in [-1,0,np.nan,np.inf]:
        with pytest.raises(ValueError):
            SynthParams(volume_depth_um=depth)


def test_rank_after_nms_removes_penalty_from_suppressed_higher_alternative():
    # Track 2 suppresses track 1's best explanation; track 1's distant
    # alternative survives. Only its rank penalty should change.
    frame=pd.DataFrame({'vanished_track_id':[2,1,1], 'survivor_track_id':[20,10,11],
                        'candidate_frame':[100,100,200], 'x_um':[0.,10.,10000.],
                        'y_um':[0.,0.,0.], 'z_um':[50000.,50000.,50000.],
                        'base_percentile':[.9,.8,.7]})
    config={'pair_mode_gap_frames':10,'maximum_modes_per_vanished_track':5,
            'coordinate_suppression':True,'lineage_log_odds_penalty':1.}
    before,_=temporal_mode_predictions(frame,seed=1,method='score',config=config)
    after,_=temporal_mode_predictions(frame,seed=1,method='score',config={**config,'lineage_rank_order':'after'})
    assert before._source_index.tolist() == after._source_index.tolist() == [0,2]
    assert before._lineage_rank.tolist() == [0,1]
    assert after._lineage_rank.tolist() == [0,0]
    assert after.score.iloc[1] == pytest.approx(.7)
    assert before.score.iloc[1] < after.score.iloc[1]
    pd.testing.assert_frame_equal(before[['candidate_frame','x_um','y_um','z_um']],
                                  after[['candidate_frame','x_um','y_um','z_um']])
