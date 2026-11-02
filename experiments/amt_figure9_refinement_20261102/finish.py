#!/usr/bin/env python3
"""Use the recovered original plot and submission relabeler on selected new cases."""
import hashlib, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
from search import HERE, REPO, ROOT, OUT, original, DELTAS, dump
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ra3d._core.track_images import point_at
sys.path.append('/home/dai-server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from pypdf import PdfReader

CHOSEN=('T22','A01')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert mpl.__version__=='3.10.9',mpl.__version__
    shortlist=json.loads((OUT/'shortlist.json').read_text())
    rows=[next(r for r in shortlist if r['review_id']==key) for key in CHOSEN]
    tracks=pd.read_parquet(OUT/'shortlist-tracks.parquet')
    specs=[];rendered={};scores={};case_records=[];panels_audit=[]
    for row in rows:
        t=int(row['candidate_frame']);key=row['kind']
        spec=original.CaseSpec(key=key,title=key,vanished_track_id=int(row['vanished_track_id']),
            survivor_track_id=int(row['survivor_track_id']),candidate_frame=t,
            display_frames=tuple(t+d for d in DELTAS),time_reference_frame=t,
            truth_event_id=int(row['matched_truth_event_id']) if key=='true_collision' else None,
            expected_shared_source_id=int(row['source_particle_id']) if key!='true_collision' else None)
        specs.append(spec)
        meta=json.loads((OUT/f"cache/{row['review_id']}.json").read_text())
        images=np.load(OUT/f"cache/{row['review_id']}.npz")
        v=tracks.loc[tracks.track_id.eq(spec.vanished_track_id)].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(spec.survivor_track_id)].sort_values('frame')
        panels=[]
        for panel in meta['panels']:
            frame=panel['frame'];delta=panel['delta']
            raw=images[f'raw_{delta}'];display=images[f'display_{delta}']
            assert raw.shape==display.shape==(128,128)
            assert np.array_equal(original.normalize_valid_crop(raw,np.ones_like(raw,dtype=bool)),display)
            vpoint=point_at(v,frame,nearest=False);spoint=point_at(s,frame,nearest=False)
            assert spoint is not None
            if delta<0 or (key!='true_collision' and delta==0):assert vpoint is not None
            if delta==4:assert vpoint is None
            panels.append({**panel,'image':display,'elapsed_ms':delta/4,'vanished_point':vpoint,'survivor_point':spoint})
            panels_audit.append({'case':key,**panel,'crop_size_px':128,'raw_array_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                'normalized_array_sha256':hashlib.sha256(display.tobytes()).hexdigest(),'actual_marker_observations':{'terminating':vpoint is not None,'continuing':spoint is not None}})
        rendered[key]=(panels,v,s)
        scores[key]=pd.Series({'static_score':row['static_score'],'full_score':row['base_score']})
        record={'case':'Ground-truth collision' if key=='true_collision' else 'Shared-particle artifact',
            'candidate_frame':t,'vanished_track_id':spec.vanished_track_id,'survivor_track_id':spec.survivor_track_id,
            'true_event_id':int(row['matched_truth_event_id']),
            'trajectory_score':float(row['static_score']),'image_feature_score':float(row['base_score']),
            'image_pre_same_fraction':float(row['pre_same_interp_fraction']),
            'final_candidate_retained':bool(row['final_candidate_retained']), 'review_id':row['review_id'],
            'display_frames':list(spec.display_frames),'display_time_ms':[d/4 for d in DELTAS]}
        if key=='true_collision':record['truth_event']=row['truth_event']
        else:record.update(shared_truth_particle_id=int(row['source_particle_id']),
                same_truth_particle_preframes=row['identity_distinct_frames'],
                terminating_truth_support_preframes=row['identity_support'][0],continuing_truth_support_preframes=row['identity_support'][1])
        case_records.append(record)
    original.TRUE_CASE,original.ARTIFACT_CASE=specs
    original.CASES=tuple(specs)
    mpl.rcdefaults();original.configure_plotting()
    fig=original.build_figure(rendered,scores,view_size=128)
    event=rows[0]['truth_event']
    subtitle=f"Ground-truth collision: {event['parent1_diameter_um']:.0f} + {event['parent2_diameter_um']:.0f} → {event['child_diameter_um']:.0f} µm"
    for ax in fig.axes:
        for label in ax.texts:
            if label.get_text().startswith('(a) Ground-truth collision:'):label.set_text('(a) '+subtitle)
    raw_pdf=OUT/'original-style-source.pdf'
    fig.savefig(raw_pdf,bbox_inches='tight',pad_inches=.025,facecolor='white',metadata={'Title':'Image-feature collision case study'})
    layout={'figsize_inches':fig.get_size_inches().tolist(),'axis_positions':[list(ax.get_position().bounds) for ax in fig.axes],
            'font':plt.rcParams['font.family'],'matplotlib':mpl.__version__,'crop_size_px':128}
    plt.close(fig)
    # Apply precisely the published paper's second-stage panel typography.
    spec=importlib.util.spec_from_file_location('prepare_figures',HERE/'source/prepare_figures.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    prepare.ROOT=OUT;prepare.register_label_font()
    manifest=json.loads((REPO.parent/'CollisionPaper/本番フォーマット/figure_sources/manifest.json').read_text())
    source_spec=next(i for i in manifest if i['output']=='fig09.pdf')
    subtitles=[subtitle,*source_spec['subtitles'][1:]]
    prepare.relabel('original-style-source.pdf',OUT/'fig09.pdf',source_spec['panel_centres'],1.0,subtitles)
    old=PdfReader(REPO.parent/'CollisionPaper/本番フォーマット/figures/fig09.pdf').pages[0]
    new=PdfReader(OUT/'fig09.pdf').pages[0]
    sizes={'original_points':[float(old.mediabox.width),float(old.mediabox.height)],
           'revised_points':[float(new.mediabox.width),float(new.mediabox.height)]}
    # Matplotlib tight bounding boxes include value labels: the new 0.998 label
    # reaches 3.16 pt farther right than the historical 0.977. The axes, image
    # panels, legends and caption centres remain at the original coordinates.
    assert abs(sizes['original_points'][1]-sizes['revised_points'][1])<.01,sizes
    assert abs(sizes['original_points'][0]-sizes['revised_points'][0])/sizes['original_points'][0]<.01,sizes
    sizes['width_note']='Tight bounding box extends for the larger score label; all original plot/panel coordinates are unchanged.'
    subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to','2400','-png',str(OUT/'fig09.pdf'),str(OUT/'fig09')],check=True)
    fig,axs=plt.subplots(2,3,figsize=(11,7))
    for r,row in enumerate(rows):
        images=np.load(OUT/f"cache/{row['review_id']}.npz")
        delta=DELTAS[0]
        raw=images[f'raw_{delta}'];wide=images[f'wide_{delta}'];valid=images[f'wide_valid_{delta}']
        panels=[original.normalize_valid_crop(wide,valid),np.rint(raw*255).astype(np.uint8),images[f'display_{delta}']]
        for ax,image,title in zip(axs[r],panels,['256 px / 1–99%','128 px / fixed 0–1','128 px / 1–99%']):
            ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        axs[r,0].set_ylabel('True collision' if r==0 else 'Shared-particle artifact')
    fig.tight_layout();fig.savefig(OUT/'crop-contrast-comparison.png',dpi=170);plt.close(fig)
    selection={
        'status':'PASS','edition':'2026-11-02 author-requested Fig.9 refinement',
        'cases':case_records,'panels':panels_audit,'layout':layout,'pdf_sizes':sizes,
        'original_paper_repository':'https://github.com/dainakai/CollisionPaper.git',
        'original_paper_commit':'2a5d3f4e4b4723aab782c2f42b473d5b5e3bbd92',
        'original_style_source':'/media/dai-server/DATAM2/siderepos/260804/CollisionPaper/scripts/plot_image_feature_case_study.py',
        'original_style_source_sha256':sha(HERE/'source/original_plot.py'),
        'submission_preparer_sha256':sha(HERE/'source/prepare_figures.py'),
        'reconstruction_config_sha256':sha(ROOT/'config/seed260745-collision.yaml'),
        'candidate_features_sha256':sha(ROOT/'data/features/validation/seed260745/paper92.parquet'),
        'saved_scores_sha256':sha(ROOT/'results/ours/candidate_scores.parquet'),
        'saved_predictions_sha256':sha(ROOT/'results/ours/predictions.parquet'),
        'crop_size_px':128,'pixel_pitch_um':10,'physical_crop_width_mm':1.28,
        'display_normalization':'Exact original plot function: linear 1st–99th percentile normalization separately for each full crop; no gamma, sharpening, denoising or local contrast.',
        'selection':'At the author request, visually inspected 30 shortlisted identity-consistent true events from 60 eligible candidates and the artifact surviving strict observation/identity checks. Selected T22 for separated pre-event particle silhouettes and an unobstructed post-event daughter. A01 has actual observations of both IDs at −1.25 ms and at 0; the continuing observation remains at +1 ms. Earlier preview alternatives are archived. This is illustrative selection after seeing results, not a performance estimate.',
        'focus':'Continuing-track smoothed depth at each displayed frame, without manual or truth-based refocusing.',
        'all_model_fits_scores_and_performance_metrics_unchanged':True,
        'final_pdf_sha256':sha(OUT/'fig09.pdf'),'final_png_sha256':sha(OUT/'fig09.png')}
    dump('selection.json',selection)
    (HERE/'results').mkdir(exist_ok=True)
    for name in ['fig09.pdf','fig09.png','crop-contrast-comparison.png','selection.json','search-protocol.json']:
        shutil.copyfile(OUT/name,HERE/'results'/name)
    print(json.dumps({'status':'PASS','chosen':CHOSEN,'scores':[[r['trajectory_score'],r['image_feature_score']] for r in case_records],
                      'pdf_sizes':sizes,'figure':str(OUT/'fig09.pdf')},indent=2))

if __name__=='__main__':main()
