#!/usr/bin/env python3
"""Assemble the author's N1/D4 selection with the original manuscript styling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from search import HERE, REPO, ROOT, original
from frame_gallery import sha
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ra3d._core.track_images import point_at
sys.path.append('/home/dai-server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from pypdf import PdfReader
import pdfplumber

OUT=REPO.parent/'audit/revision-figure9-final-20261103'
A=REPO.parent/'audit/revision-figure9-confirmed-a-20261103'
A_CACHE=REPO.parent/'audit/revision-figure9-interior-20261103/N1/cache'
B=REPO.parent/'audit/revision-figure9-shared-particle-20261103'
PAPER=REPO.parent/'CollisionPaper/本番フォーマット'


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    assert mpl.__version__=='3.10.9'
    a=json.loads((A/'selection.json').read_text())
    review=json.loads((B/'review.json').read_text())
    b=next(r for r in review['cases'] if r['case_id']=='D4')
    assert a['status']=='author_confirmed' and a['display_frames']==[3957,3958,3959]
    assert b['candidate_key']==[17945,18031,5337] and b['selected_frames']==[5335,5337,5339]
    assert sha(A/'fig09-a-confirmed.pdf')==a['pdf_sha256']
    protected={str(A/'fig09-a-confirmed.pdf'):sha(A/'fig09-a-confirmed.pdf'),
        str(A/'selection.json'):sha(A/'selection.json'),str(PAPER/'template.tex'):sha(PAPER/'template.tex')}
    assert protected[str(PAPER/'template.tex')]=='1fbbcfcf35f5011067d92429359fd7785f98653ace260b49622b32f60e1e83b5'
    frozen=json.loads((REPO/'experiments/amt_full_20261101/results/manifest.json').read_text())['numerical_outputs_manifest']
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    features=pd.read_parquet(ROOT/'data/features/validation/seed260745/paper92.parquet').set_index(['vanished_track_id','survivor_track_id','candidate_frame'])
    scores=pd.read_parquet(ROOT/'results/ours/candidate_scores.parquet').set_index(['vanished_track_id','survivor_track_id','candidate_frame'])
    predictions=pd.read_parquet(ROOT/'results/ours/predictions.parquet')
    final=set(zip(predictions.vanished_track_id,predictions.survivor_track_id,predictions.candidate_frame_raw))
    configs=[dict(id='N1',key='true_collision',t=3958,v=12754,s=12764,deltas=[-1,0,1],
                  panel_meta=a['panels'],cache=A_CACHE,cache_ids=[4,5,6],truth_event=a['truth_event'],
                  trajectory_score=a['trajectory_score'],image_feature_score=a['image_feature_score']),
             dict(id='D4',key='shared_particle_artifact',t=5337,v=17945,s=18031,deltas=[-2,0,2],
                  panel_meta=b['selected_panels'],cache=B/'D4/cache',cache_ids=[4,5,6],truth_event=None,
                  trajectory_score=b['trajectory_score'],image_feature_score=b['image_feature_score'])]
    specs=[];reconstructed={};candidate_rows={};cases=[];panels_audit=[];identities=[]
    for c in configs:
        spec=original.CaseSpec(key=c['key'],title=c['key'],vanished_track_id=c['v'],survivor_track_id=c['s'],
            candidate_frame=c['t'],display_frames=tuple(c['t']+d for d in c['deltas']),time_reference_frame=c['t'],
            truth_event_id=1204 if c['id']=='N1' else None,expected_shared_source_id=7364 if c['id']=='D4' else None)
        specs.append(spec)
        key=(c['v'],c['s'],c['t']);frow=features.loc[key]
        assert float(frow.static_score)==c['trajectory_score'] and float(scores.loc[key].base_score)==c['image_feature_score']
        assert int(frow.image_enriched)==1 and int(frow.rendered_pre_frames)==9
        assert (key in final)==(c['id']=='N1')
        v=tracks.loc[tracks.track_id.eq(c['v'])].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(c['s'])].sort_values('frame')
        panels=[];obs=[]
        for delta,meta,cache_id in zip(c['deltas'],c['panel_meta'],c['cache_ids'],strict=True):
            frame=c['t']+delta;assert meta['frame']==frame
            saved=np.load(c['cache']/f'{cache_id}.npz');raw,valid=saved['raw'],saved['valid']
            display=original.normalize_valid_crop(raw,valid)
            assert valid.all() and raw.shape==display.shape==(128,128)
            if c['id']=='N1':
                assert hashlib.sha256(raw.tobytes()).hexdigest()==meta['raw_array_sha256']
                assert hashlib.sha256(display.tobytes()).hexdigest()==meta['normalized_array_sha256']
            sp=point_at(s,frame,nearest=False);vp=point_at(v,frame,nearest=False)
            assert sp is not None
            assert (vp is not None)==(delta<0 or (c['id']=='D4' and delta==0))
            assert np.allclose(saved['center'],[sp.x,sp.y,sp.z])
            x0,y0=int(saved['x0']),int(saved['y0'])
            assert (x0,y0)==(round(sp.x)-64,round(sp.y)-64)
            assert (x0,y0)==(meta['x0'],meta['y0'])
            panels.append(dict(frame=frame,elapsed_ms=delta/4,image=display,x0=x0,y0=y0,
                vanished_point=vp,survivor_point=sp))
            panels_audit.append(dict(case=c['id'],frame=frame,delta=delta,time_ms=delta/4,x0=x0,y0=y0,
                center_x=float(sp.x),center_y=float(sp.y),center_z=float(sp.z),recorded_fraction=1.,
                crop_hologram_margin_px=min(x0,y0,1024-x0-128,1024-y0-128),
                cache_path=str(c['cache']/f'{cache_id}.npz'),
                raw_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                normalized_array_sha256=hashlib.sha256(display.tobytes()).hexdigest(),
                actual_marker_observations=dict(terminating=vp is not None,continuing=True)))
            gt=pd.read_parquet(ROOT/'data/labels/validation/seed260745/particles.parquet',filters=[('frame','=',frame)])
            for tid,p in [(c['v'],vp),(c['s'],sp)]:
                if p is None:continue
                expected=7364 if c['id']=='D4' else (5288 if tid==12754 else 5289) if delta<0 else 5377
                q=gt.loc[gt.particle_id.eq(expected)].iloc[0]
                distances=np.hypot(gt.x_px-p.x,gt.y_px-p.y)
                xy=float(np.hypot(q.x_px-p.x,q.y_px-p.y));dz=float((p.z-1)*100-q.z_um)
                assert int(gt.loc[distances.idxmin()].particle_id)==expected
                assert xy<=q.diameter_um/20 and abs(dz)<100
                clearance=min(q.x_px-x0,x0+127-q.x_px,q.y_px-y0,y0+127-q.y_px)-q.diameter_um/20
                assert clearance>0
                obs.append(dict(frame=frame,track_id=tid,truth_particle_id=expected,track_xy_px=[float(p.x),float(p.y)],
                    truth_xy_px=[float(q.x_px),float(q.y_px)],truth_diameter_um=float(q.diameter_um),
                    xy_error_px=xy,depth_error_um=dz,truth_disk_clearance_px=float(clearance),nearest_truth_center_is_expected=True))
        reconstructed[spec.key]=(panels,v,s)
        candidate_rows[spec.key]=pd.Series(dict(static_score=c['trajectory_score'],full_score=c['image_feature_score']))
        record=dict(case='Ground-truth collision' if c['id']=='N1' else 'Shared-particle artifact',review_id=c['id'],
            candidate_frame=c['t'],vanished_track_id=c['v'],survivor_track_id=c['s'],true_event_id=1204 if c['id']=='N1' else -1,
            trajectory_score=c['trajectory_score'],image_feature_score=c['image_feature_score'],
            image_pre_same_fraction=float(frow.pre_same_interp_fraction),final_candidate_retained=key in final,
            display_frames=list(spec.display_frames),display_time_ms=[d/4 for d in c['deltas']],author_confirmed=True)
        if c['truth_event'] is not None:record['truth_event']=c['truth_event']
        else:
            record.update(shared_truth_particle_id=7364,shared_truth_diameter_um=b['truth_diameter_um'],
                score_decrease=c['trajectory_score']-c['image_feature_score'],
                diagnostic_note=b['interpretation']['caveat'],particle_history=b['interpretation']['history'],
                source_particle_creation_frame=5312,continuing_track_start_frame=int(s.frame.min()))
            assert int(s.frame.min())==5319
            events=pd.read_parquet(ROOT/'data/labels/validation/seed260745/collision_events.parquet')
            near=events.loc[(events.frame-c['t']).abs().le(5)]
            assert not np.hypot(near.x_um-frow.x_um,near.y_um-frow.y_um).le(350).any()
        cases.append(record);identities.append(dict(review_id=c['id'],observations=obs))
    assert sum(len(c['observations']) for c in identities)==9
    original.TRUE_CASE,original.ARTIFACT_CASE=specs;original.CASES=tuple(specs)
    mpl.rcdefaults();original.configure_plotting()
    fig=original.build_figure(reconstructed,candidate_rows,view_size=128)
    # Keep the original physical layout, but embed all 128 source pixels in
    # the PDF. Default savefig DPI otherwise downsamples these panels to 106px.
    # Colormap and plotted coordinates stay exactly those of the original.
    for ax,panel in zip(fig.axes[:6],panels_audit,strict=True):
        artist=ax.images[0];artist.set_interpolation('none')
        rgb=artist.to_rgba(artist.get_array(),bytes=True)[:,:,:3]
        # Matplotlib stores native PDF image rows bottom-up and applies the
        # corresponding page transformation; PNG rendering verifies orientation.
        panel['pdf_rgb_array_sha256']=hashlib.sha256(np.flipud(rgb).tobytes()).hexdigest()
    e=a['truth_event'];subtitle=f"Ground-truth collision: {e['parent1_diameter_um']:.0f} + {e['parent2_diameter_um']:.0f} → {e['child_diameter_um']:.0f} µm"
    for ax in fig.axes:
        for label in ax.texts:
            if label.get_text().startswith('(a) Ground-truth collision:'):label.set_text('(a) '+subtitle)
    source_pdf=OUT/'original-style-source.pdf'
    fig.savefig(source_pdf,bbox_inches='tight',pad_inches=.025,facecolor='white',metadata={'Title':'Fig.9: author-confirmed N1 and D4'})
    layout=dict(figsize_inches=fig.get_size_inches().tolist(),axis_positions=[list(ax.get_position().bounds) for ax in fig.axes])
    plt.close(fig)
    mod=importlib.util.spec_from_file_location('prepare_final_fig9',HERE/'source/prepare_figures.py')
    prepare=importlib.util.module_from_spec(mod);mod.loader.exec_module(prepare)
    prepare.ROOT=OUT;prepare.register_label_font()
    source_spec=next(s for s in json.loads((PAPER/'figure_sources/manifest.json').read_text()) if s['output']=='fig09.pdf')
    prepare.relabel('original-style-source.pdf',OUT/'fig09.pdf',source_spec['panel_centres'],1.,[subtitle,*source_spec['subtitles'][1:]])
    subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to','2400','-png',str(OUT/'fig09.pdf'),str(OUT/'fig09')],check=True)
    geometry=[];used_fonts=set()
    for path in [PAPER/'figures/fig09.pdf',OUT/'fig09.pdf']:
        with pdfplumber.open(path) as pdf:
            page=pdf.pages[0];boxes=sorted(page.images,key=lambda box:(round(box['top']),box['x0']))
            assert len(boxes)==6
            # Compare the visible panel borders, not the clipped image XObject
            # extent: the old 106px raster extends 0.59pt beyond its clip box.
            borders=[line for line in page.lines if abs(line['linewidth']-.6)<1e-8
                and 70<max(line['width'],line['height'])<80]
            assert len(borders)==24
            geometry.append([[line[k] for k in ['x0','x1','top','bottom']] for line in borders])
            assert all(-.1<=p['x0']<=p['x1']<=page.width+.1 and -.1<=p['top']<=p['bottom']<=page.height+.1 for p in page.chars)
            if path==OUT/'fig09.pdf':used_fonts={p['fontname'] for p in page.chars}
    assert np.allclose(*geometry,atol=.01),geometry
    fonts=subprocess.check_output(['/usr/bin/pdffonts',str(OUT/'fig09.pdf')],text=True)
    for name in used_fonts:
        rows=[r for r in fonts.splitlines()[2:] if r.split()[0]==name]
        assert len(rows)==1 and ' yes ' in rows[0],(name,rows)
    images=PdfReader(OUT/'fig09.pdf').pages[0].images
    assert len(images)==6
    assert all(im.image.size==(128,128) for im in images)
    assert {hashlib.sha256(im.image.convert('RGB').tobytes()).hexdigest() for im in images}=={p['pdf_rgb_array_sha256'] for p in panels_audit}
    write('identity-audit.json',dict(status='PASS',cases=identities,scope='Displayed observations only; no training-label or evaluation changes.'))
    write('search-protocol.json',dict(status='PASS',author_confirmed=['N1','D4'],
        source_review_sha256=sha(B/'review.json'),source_a_selection_sha256=sha(A/'selection.json'),
        selection='Post-hoc illustrative visual selection confirmed by the author; not an estimate of the mean image-feature effect.',
        rejected_distinct_particle_examples=['B01','B02'],candidate_review='fig09-artifact-options.html',
        diagnosis='D4 frames -0.75/-0.25ms have red positions outside the truth disk; +0.25ms has no blue observation. They remain visible as diagnostics in the candidate review, not in the paper figure.'))
    # Retain earlier provenance as history, never as a current C04/H01 choice.
    history=json.loads((REPO.parent/'audit/revision-figure9-centered-20261102/original-comparison.json').read_text())
    history['historical_selected_cases']=history.pop('selected_cases')
    history['selected_cases']=['N1','D4']
    history['historical_display_frames_relative']=history.pop('display_frames_relative')
    history['display_frames_relative']={'N1':[-1,0,1],'D4':[-2,0,2]}
    history['historical_display_time_note']=history.pop('display_time_note')
    history['display_time_note']='Author-confirmed N1: -0.25/0/+0.25ms; D4: -0.50/0/+0.50ms. All six centered crops are fully recorded.'
    history['same_planned_slot_1165']['decision']='The original planned slot was inspected; the author ultimately confirmed N1, true event 1204, and D4, same-particle artifact.'
    history['scope_note']='Original regeneration comparison remains valid. Other notes about C04/H01 are historical; current selection and times are in selection.json.'
    write('original-comparison.json',history)
    selection=dict(status='PASS',author_confirmed=True,edition='2026-11-03 author-confirmed N1/D4',cases=cases,panels=panels_audit,
        layout=layout,crop_size_px=128,pixel_pitch_um=10,phase_iterations=3,fully_recorded_panels=6,
        display_normalization='Exact original 1st/99th percentile normalization of recorded pixels; no manual focus, centering correction, gamma or sharpening.',
        crop_position='128 pixels centered on actual continuing-track positions; all six crops entirely inside the hologram, without clamping.',
        pdf_image_embedding='Native 128x128 pixels with the unchanged original gray colormap, replacing default savefig downsampling to 106x106. Original physical image-panel geometry is unchanged.',
        focus='Continuing-track saved smoothed Z; all final panels have actual continuing-track observations.',
        final_pdf_sha256=sha(OUT/'fig09.pdf'),final_png_sha256=sha(OUT/'fig09.png'),
        original_style_source_sha256=sha(HERE/'source/original_plot.py'),submission_preparer_sha256=sha(HERE/'source/prepare_figures.py'),
        candidate_features_sha256=sha(ROOT/'data/features/validation/seed260745/paper92.parquet'),
        saved_scores_sha256=sha(ROOT/'results/ours/candidate_scores.parquet'),saved_predictions_sha256=sha(ROOT/'results/ours/predictions.parquet'),
        reconstruction_config_sha256=sha(ROOT/'config/seed260745-collision.yaml'),
        all_model_fits_scores_and_performance_metrics_unchanged=True,
        selection='Author confirmed (a) N1 at -0.25/0/+0.25ms and accepted the recommended (b) D4 at -0.50/0/+0.50ms.',
        protected_original_inputs=protected)
    write('selection.json',selection)
    assert all(sha(p)==h for p,h in protected.items()) and all(sha(ROOT/p)==h for p,h in frozen.items())
    write('verification.json',dict(status='PASS',visual_review='pending',
        final_pdf_sha256=sha(OUT/'fig09.pdf'),selection_sha256=sha(OUT/'selection.json'),
        identity_audit_sha256=sha(OUT/'identity-audit.json'),original_comparison_sha256=sha(OUT/'original-comparison.json'),
        frozen_evaluation_files_unchanged=len(frozen),original_image_panel_geometry_identical=True,
        used_fonts_embedded_and_no_clipped_text=True,used_fonts=sorted(used_fonts),
        six_pdf_images_match_reviewed_crops_through_original_colormap=True,native_pdf_image_size_px=128,actual_observations_checked=9,
        fully_recorded_panels=6,manuscript_unchanged=True,confirmed_a_pixels_unchanged=True))
    target=HERE/'results/final';target.mkdir(exist_ok=True)
    for name in ['fig09.pdf','fig09.png','selection.json','identity-audit.json','search-protocol.json','original-comparison.json','verification.json']:
        shutil.copyfile(OUT/name,target/name)
    print(json.dumps({'status':'PASS','author_confirmed':['N1','D4'],'figure':str(OUT/'fig09.pdf'),
        'panels':6,'actual_observations':9,'visual_review':'pending'},indent=2))


if __name__=='__main__':main()
