"""Generate five display crops per human-confirmed Fig.10 candidate with the frozen engine."""
from case_images import main
from gallery import SELECTION
from common import *
import pandas as pd

if __name__=='__main__':
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet').set_index('candidate_id')
    cases=[(f'{i:02d}',cid,int(labels.loc[cid,'candidate_frame_raw'])) for i,(cid,_) in enumerate(SELECTION,1)]
    main(cases,folder='gallery_images',offsets=(-8,-4,0,4,8),
         origin_policy='Raw candidate frame. Fixed display offsets -8,-4,0,+4,+8 frames; no redefinition of the analysis time.',
         crop_center_policy='observed_parent_bounds')
