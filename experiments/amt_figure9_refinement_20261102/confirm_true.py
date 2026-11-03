#!/usr/bin/env python3
"""Export the author-selected Fig.9(a) component while (b) is being reviewed."""
import hashlib
import json
import shutil
import subprocess

from search import REPO, ROOT, HERE, original
from frame_gallery import FINAL, sha
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ra3d._core.track_images import point_at

SOURCE = REPO.parent/'audit/revision-figure9-interior-20261103'
OUT = REPO.parent/'audit/revision-figure9-confirmed-a-20261103'
OUT.mkdir(exist_ok=True)


def main():
    assert mpl.__version__ == '3.10.9'
    before={str(p):sha(p) for p in [FINAL/'fig09.pdf',FINAL/'selection.json',
        REPO.parent/'audit/revision-paper-replacements-20261102/paper-replacements.zip']}
    row=next(c for c in json.loads((SOURCE/'rendered-cases.json').read_text()) if c['case_id']=='N1')
    r=row['source'];t=int(r['candidate_frame'])
    assert (t,int(r['vanished_track_id']),int(r['survivor_track_id']))==(3958,12754,12764)
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    v=tracks.loc[tracks.track_id.eq(r['vanished_track_id'])].sort_values('frame')
    s=tracks.loc[tracks.track_id.eq(r['survivor_track_id'])].sort_values('frame')
    panels=[];evidence=[]
    (OUT/'images').mkdir(exist_ok=True);(OUT/'tracks').mkdir(exist_ok=True)
    for i,d in zip([4,5,6],[-1,0,1]):
        source_panel=next(p for p in row['panels'] if p['delta']==d)
        saved=np.load(SOURCE/'N1/cache'/f'{i}.npz')
        raw,valid=saved['raw'],saved['valid']
        display=original.normalize_valid_crop(raw,valid)
        assert valid.all()
        f=t+d;sp=point_at(s,f,nearest=False);vp=point_at(v,f,nearest=False)
        assert sp is not None and (vp is not None)==(d==-1)
        assert source_panel['x0']==round(sp.x)-64 and source_panel['y0']==round(sp.y)-64
        panels.append({**source_panel,'image':display,'elapsed_ms':d/4,'vanished_point':vp,'survivor_point':sp})
        evidence.append({**source_panel,'raw_array_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
            'normalized_array_sha256':hashlib.sha256(display.tobytes()).hexdigest(),
            'actual_marker_observations':{'terminating':vp is not None,'continuing':True}})
        for kind in ['images','tracks']:
            shutil.copyfile(SOURCE/'N1'/kind/source_panel['filename'],OUT/kind/source_panel['filename'])
    e=r['truth_event']
    spec=original.CaseSpec(key='true_collision',title='Ground-truth collision',
        vanished_track_id=12754,survivor_track_id=12764,candidate_frame=t,
        display_frames=(3957,3958,3959),time_reference_frame=t,
        truth_event_id=int(e['collision_event_id']),expected_shared_source_id=None)
    original.TRUE_CASE=spec
    original.CASES=(spec,)
    # Use the original full-figure layout, then remove its score axis. The sole
    # remaining image row retains original panel dimensions, fonts and markers.
    old_b=json.loads((FINAL/'selection.json').read_text())['cases'][1]
    scores={spec.key:pd.Series({'static_score':r['static_score'],'full_score':r['base_score']}),
        original.ARTIFACT_CASE.key:pd.Series({'static_score':old_b['trajectory_score'],'full_score':old_b['image_feature_score']})}
    mpl.rcdefaults();original.configure_plotting()
    fig=original.build_figure({spec.key:(panels,v,s)},scores,view_size=128)
    assert len(fig.axes)==4
    fig.delaxes(fig.axes[-1])
    subtitle=f"(a) Ground-truth collision: {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} → {e['child_diameter_um']:.0f} µm"
    labels=[label for ax in fig.axes for label in ax.texts if label.get_text().startswith('(a) Ground-truth collision:')]
    assert len(labels)==1
    labels[0].set_text(subtitle)
    geometry=[list(ax.get_position().bounds) for ax in fig.axes]
    pdf=OUT/'fig09-a-confirmed.pdf'
    fig.savefig(pdf,bbox_inches='tight',pad_inches=.025,facecolor='white',metadata={'Title':'Fig.9(a): author-confirmed N1, -0.25 / 0 / +0.25 ms'})
    plt.close(fig)
    subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to','2000','-png',str(pdf),str(OUT/'fig09-a-confirmed')],check=True)
    obs=[p for p in r['observations'] if p['delta'] in [-1,0,1]]
    assert len(obs)==4 and all(p['one_to_one_mapping_present'] for p in obs)
    decision=dict(status='author_confirmed',case_id='N1',candidate_frame=t,vanished_track_id=12754,
        survivor_track_id=12764,display_frames=[3957,3958,3959],display_time_ms=[-.25,0,.25],
        display_labels=['N1-4','N1-5','N1-6'],truth_event=e,
        trajectory_score=r['static_score'],image_feature_score=r['base_score'],
        panels=evidence,observations=obs,actual_markers=4,all_crops_fully_recorded=True,
        minimum_crop_hologram_margin_px=min(p['hologram_margin_px'] for p in evidence),
        original_panel_positions=geometry,original_plot_sha256=sha(HERE/'source/original_plot.py'),
        track_sha256=sha(ROOT/'runs/tracking/seed260745/smoothed.parquet'),
        phase_iterations=3,crop_size_px=128,pixel_pitch_um=10,
        pdf_sha256=sha(pdf),png_sha256=sha(OUT/'fig09-a-confirmed.png'),
        b_status='New interior candidates under author review. Full Fig.9 and author replacement ZIP await b selection.',
        old_complete_figure_and_zip_unchanged=before)
    assert before=={p:sha(p) for p in before}
    (OUT/'selection.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
    target=HERE/'results/confirmed-a';target.mkdir(parents=True,exist_ok=True)
    for name in ['fig09-a-confirmed.pdf','fig09-a-confirmed.png','selection.json']:
        shutil.copyfile(OUT/name,target/name)
    print(json.dumps({'status':'PASS','selected':'N1','frames':[3957,3958,3959],
        'times_ms':[-.25,0,.25],'actual_markers':4,'pdf':str(pdf)},indent=2))


if __name__=='__main__':main()
