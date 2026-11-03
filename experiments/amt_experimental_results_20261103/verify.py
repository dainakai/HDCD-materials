"""Independent accounting checks and a bounded artifact/link audit."""
import sqlite3
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import numpy as np
import pandas as pd
from pypdf import PdfReader
from PIL import Image
from common import *

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ['href','src']:
            if key in a:self.links.append(a[key])
        if 'id' in a:self.ids.add(a['id'])

def main():
    checks=[]
    def passed(name):checks.append(name)
    frozen=read(PRIVATE/'freeze.json')
    assert sha(PRIVATE/'reviews.sqlite3')==frozen['snapshot_sha256']
    assert sha(PRIVATE/'reviewed_candidates.parquet')==frozen['joined_sha256']
    labels=pd.read_parquet(PRIVATE/'reviewed_candidates.parquet')
    with sqlite3.connect(f'file:{CAMPAIGN}/reviews.sqlite3?mode=ro',uri=True) as db:
        live=pd.read_sql_query('SELECT candidate_id,label,updated_at,asset_sha256 FROM reviews ORDER BY candidate_id',db)
    expected=labels[['candidate_id','label','updated_at','asset_sha256']].sort_values('candidate_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(live,expected)
    passed('Frozen SQLite snapshot and live review labels agree; 1120 model/prediction/asset-bound judgments retained')
    curve=pd.read_csv(DATA/'review_budget.csv')
    for r in curve.itertuples():
        subset=labels.loc[labels.review_rank.le(r.top_n_per_scene)]
        assert len(subset)==r.candidates_reviewed==14*r.top_n_per_scene
        assert subset.label.eq('collision').sum()==r.cumulative_confirmed_collisions
        assert subset.label.eq('non_collision').sum()==r.cumulative_non_collisions
        assert subset.label.eq('ambiguous').sum()==r.cumulative_ambiguous
        np.testing.assert_allclose(r.top_n_precision,subset.label.eq('collision').mean())
    assert curve.loc[curve.coverage_of_reviewed_confirmed.ge(.95),'top_n_per_scene'].iloc[0]==34
    passed('All 80 review-budget points independently recounted from the frozen labels, including ambiguous denominators')
    boundaries=read(DATA/'figure12_thresholds.json')
    assert not np.isclose(curve.top_n_precision,.5,rtol=0,atol=1e-12).any()
    assert curve.loc[curve.top_n_precision.ge(.5),'top_n_per_scene'].max()==24
    for key,n in [('last_precision_at_least_50',24),('next_below_50',25),('first_coverage_at_least_95',34)]:
        point=boundaries[key];selected=labels.loc[labels.review_rank.le(n)]
        assert point['per_sequence']==n and point['total_reviews']==len(selected)
        assert point['confirmed']==selected.label.eq('collision').sum()
        np.testing.assert_allclose(point['precision'],selected.label.eq('collision').mean())
        np.testing.assert_allclose(point['coverage'],selected.label.eq('collision').sum()/186)
    pdf12=PdfReader(FIGURES/'fig12.pdf').pages[0].extract_text()
    assert all(value in pdf12 for value in ['51.49','49.43','93.01','95.16','37.18'])
    assert 'rounded' not in (PACKAGE/'text/caption_fig12.tex').read_text()
    passed('Requested Fig.12 discrete boundaries independently verified: N=24/25, first 95% coverage at N=34; no fractional budget invented')
    counts=pd.read_csv(DATA/'review_counts_by_scene.csv')
    assert counts[['collision','non_collision','ambiguous']].sum().tolist()==[186,922,12]
    pairs=pd.read_csv(DATA/'diameter_pair_counts.csv')
    positive=labels.loc[labels.label.eq('collision')]
    d=positive[['pre_diameter_vanished_um','pre_diameter_survivor_um']].to_numpy(float)
    valid=np.isfinite(d).all(axis=1)&(d>0).all(axis=1)
    assert valid.sum()==185
    ordered=np.sort(d[valid],axis=1)
    independent=np.zeros((11,11),int)
    for small,large in ordered:
        assert 25<=small<=large<300
        independent[int((small-25)//25),int((large-25)//25)]+=1
    for r in pairs.itertuples():assert independent[r.small_bin_index,r.large_bin_index]==r.confirmed_collision_lower_bound
    assert independent.sum()==185
    passed('185 parent pairs independently assigned from two valid parent diameters; the missing-parent event is retained only in total counts')
    scenes=pd.read_csv(DATA/'scene_collision_counts.csv',float_precision='round_trip')
    assert scenes.iloc[-1].confirmed_collision_lower_bound==186
    np.testing.assert_allclose(scenes.iloc[-1].predicted_expected_count,scenes.iloc[:-1].predicted_expected_count.sum(),rtol=1e-13)
    np.testing.assert_allclose(pairs.predicted_expected_count.sum(),scenes.iloc[-1].predicted_expected_count,rtol=1e-13)
    np.testing.assert_allclose(scenes.predicted_total_sd**2,scenes.predicted_expected_count,rtol=1e-13)
    np.testing.assert_allclose(scenes.iloc[-1].exposure_volume_m3_s,14*(1024*10e-6)**2*(1023*100e-6)*(10917/4000),rtol=1e-13)
    passed('Full-recording nominal-volume exposure, scene/pair totals and square-root Poisson pooling agree')
    physical=pd.read_csv(DATA/'physical_consistency.csv')
    assert len(physical)==1108
    assert physical.groupby('classification_role').diameter_available.sum().to_dict()=={'FP':908,'TP':184}
    assert physical.groupby('classification_role').velocity_available.sum().to_dict()=={'FP':822,'TP':179}
    for role in ['TP','FP']:
        for quantity,x,y,avail in [('diameter_um','expected_post_diameter_um','measured_post_diameter_um','diameter_available'),
                                  ('velocity_m_s','expected_post_vy_m_s','measured_post_vy_m_s','velocity_available')]:
            subset=physical.loc[physical.classification_role.eq(role)&physical[avail]]
            actual=float(np.sqrt(np.mean((subset[y]-subset[x])**2)))
            metric=pd.read_csv(DATA/'physical_error_summary.csv').query('version == "revision" and role == @role and quantity == @quantity').iloc[0]
            np.testing.assert_allclose(actual,metric.rmse,rtol=1e-12)
    passed('Physical diagnostic sample counts and all published RMSE values independently checked from exported figure data')
    eligibility=pd.read_csv(DATA/'figure13_eligibility_all1120.csv').set_index('candidate_id')
    source=pd.read_parquet(PRIVATE/'physical_events_all.parquet').set_index('candidate_id').loc[eligibility.index]
    assert len(eligibility)==1120 and eligibility.index.is_unique
    for panel,field in [('a','diameter_available'),('b','velocity_available')]:
        np.testing.assert_array_equal(eligibility['fig13'+panel+'_included'],source.status.ne('ambiguous')&source[field])
    excluded=pd.read_csv(DATA/'figure13_excluded_samples.csv')
    assert set(excluded.candidate_id)==set(eligibility.index[~eligibility.fig13a_included|~eligibility.fig13b_included])
    assert excluded.status.value_counts().to_dict()=={'non_collision':100,'ambiguous':12,'collision':7}
    evidence=pd.read_csv(DATA/'figure13_window_evidence.csv')
    assert len(evidence)==3360 and not evidence.duplicated(['candidate_id','role']).any()
    for w in evidence.itertuples():
        frames=[] if pd.isna(w.velocity_frame_list) else [int(f) for f in str(w.velocity_frame_list).split(';')]
        assert len(frames)==w.velocity_observed_points
        assert w.velocity_api_points==(len(frames) if len(frames)>=2 else 0)
        assert frames==sorted(set(frames))
        if frames:assert min(frames)>=w.velocity_start_frame and max(frames)<=w.velocity_end_frame
    assert evidence.raw_and_smoothed_frames_identical.all()
    assert evidence.diameter_invalid_points.sum()==evidence.velocity_invalid_y_points.sum()==0
    passed('All 1120 eligibility decisions reconcile with Fig.13; every one of 119 exclusions has frame-level source evidence; true one-point windows distinguished from API zero')
    h=pd.read_csv(DATA/'score_histogram.csv')
    assert h.groupby('status')['count'].sum().to_dict()=={'collision':186,'non_collision':922}
    passed('Histogram bins contain every binary-labeled candidate without clipping')
    figures=[4,5,10,11,12,13,14,15]
    for n in figures:
        reader=PdfReader(FIGURES/f'fig{n:02d}.pdf');assert len(reader.pages)==1
        box=reader.pages[0].mediabox;assert float(box.width)>250 and float(box.height)>100
        assert len(reader.pages[0].extract_text())>40
        for subdir in ['figures','preview']:
            with Image.open(PACKAGE/subdir/f'fig{n:02d}.png') as im:assert min(im.size)>350
    assert len(PdfReader(PACKAGE/'experimental_figures.pdf').pages)==8
    passed('Eight single-page figure PDFs, high-resolution PNGs and the eight-page combined PDF parse successfully')
    gallery=read(DATA/'figure10_candidates_provenance.json')
    assert len(gallery['cases'])==10 and len(PdfReader(PACKAGE/'fig10-candidates.pdf').pages)==10
    label_map=labels.set_index('candidate_id')
    for c in gallery['cases']:
        event=label_map.loc[c['candidate_id']]
        assert event.status=='collision' and event.model_sha256==MODEL_SHA
        assert c['display_t0_frame']==c['candidate_frame_raw']==event.candidate_frame_raw
        assert c['crop_center_policy']=='observed_parent_bounds' and c['phase_iterations']==3
        assert sha(PACKAGE/'gallery'/f'{c["case"]}.gif')==event.asset_sha256
        assert len(PdfReader(PACKAGE/'gallery'/f'{c["case"]}.pdf').pages)==1
        tracks=pd.read_parquet(CAMPAIGN/f'scenes/scene{c["scene_num"]:04d}/tracking/smoothed.parquet')
        for frame in c['frames']:
            assert sha(PRIVATE/'gallery_images'/frame['cache'])==frame['sha256']
            points=tracks.loc[tracks.track_id.isin([event.vanished_track_id,event.survivor_track_id])&tracks.frame.eq(frame['frame'])]
            for point in points.itertuples():
                radius=point.diameter_px/2
                assert 0<=point.x-frame['x0']-radius and point.x-frame['x0']+radius<128
                assert 0<=point.y-frame['y0']-radius and point.y-frame['y0']+radius<128
        assert [r['frame']-c['candidate_frame_raw'] for r in c['frames']]==[-8,-4,0,4,8]
    passed('Ten human-confirmed illustration cases preserve candidate/model/GIF identity; all observed parent extents fit their 50 reconstructed crops')
    pages={name:Links() for name in ['index.html','fig10-candidates.html','fig13-exclusions.html']}
    for name,parser in pages.items():parser.feed((PACKAGE/name).read_text())
    scheduled={'verification.json','experimental-replacements.zip'}
    for name,parser in pages.items():
        for link in parser.links:
            url=urlsplit(link)
            assert not url.scheme and not url.netloc,'External resource in report'
            if url.path:
                assert '..' not in Path(unquote(url.path)).parts
                assert (PACKAGE/unquote(url.path)).is_file() or url.path in scheduled,link
                if url.fragment and url.path in pages:assert url.fragment in pages[url.path].ids
            elif url.fragment:assert url.fragment in parser.ids,link
        content=(PACKAGE/name).read_text()
        assert 'style=' not in content and '<script' not in content
    passed('All three report pages, gallery GIF/crop links, exclusion evidence, figure downloads and navigation anchors resolve without external assets')
    manuscript=WORK/'CollisionPaper/本番フォーマット/template.tex'
    assert sha(manuscript)==read(PACKAGE/'text/replacement_anchors.json')['manuscript_sha256_at_build']
    passed('Original manuscript remains unmodified')
    snapshot_paths=[p.name for p in PACKAGE.rglob('*') if p.is_file()]
    assert not any('sqlite' in p or 'review-access' in p or p.endswith(('.pt','.pth')) for p in snapshot_paths)
    passed('No review database, review access configuration or model weights in the public package')
    save(PACKAGE/'verification.json',{'status':'PASS','verified_at':stamp(),'checks':checks,
        'visual_review':'Existing eight figures previously inspected. Updated Fig.12 and all ten new case PDFs rendered and inspected; observed-parent crop bounds also checked numerically.',
        'limitations':['No claim of absolute experimental recall.','New and public cohorts differ; comparisons are not causal ablations.',
            'No terminal TeX compiler is installed; standalone TeX integration is supplied but the original manuscript is not compiled or edited.']})
    print('PASS:',len(checks),'independent accounting/artifact checks')

if __name__=='__main__':main()
