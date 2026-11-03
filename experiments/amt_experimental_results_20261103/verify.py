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
    parser=Links();parser.feed((PACKAGE/'index.html').read_text())
    scheduled={'verification.json','experimental-replacements.zip'}
    for link in parser.links:
        url=urlsplit(link)
        assert not url.scheme and not url.netloc,'External resource in report'
        if url.path:
            assert '..' not in Path(unquote(url.path)).parts
            assert (PACKAGE/unquote(url.path)).is_file() or url.path in scheduled,link
        elif url.fragment:assert url.fragment in parser.ids,link
    content=(PACKAGE/'index.html').read_text()
    assert 'style=' not in content and '<script' not in content
    passed('All local report assets, figure links, caption links and navigation anchors resolve; no inline scripts/styles or external assets')
    manuscript=WORK/'CollisionPaper/本番フォーマット/template.tex'
    assert sha(manuscript)==read(PACKAGE/'text/replacement_anchors.json')['manuscript_sha256_at_build']
    passed('Original manuscript remains unmodified')
    snapshot_paths=[p.name for p in PACKAGE.rglob('*') if p.is_file()]
    assert not any('sqlite' in p or 'review-access' in p or p.endswith(('.pt','.pth')) for p in snapshot_paths)
    passed('No review database, review access configuration or model weights in the public package')
    save(PACKAGE/'verification.json',{'status':'PASS','verified_at':stamp(),'checks':checks,
        'visual_review':'All eight figures rendered from the final PDFs and inspected; Fig.10 frame context and Fig.13 outliers retained.',
        'limitations':['No claim of absolute experimental recall.','New and public cohorts differ; comparisons are not causal ablations.',
            'No terminal TeX compiler is installed; standalone TeX integration is supplied but the original manuscript is not compiled or edited.']})
    print('PASS:',len(checks),'independent accounting/artifact checks')

if __name__=='__main__':main()
