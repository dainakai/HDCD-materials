"""Make an atomic SQLite backup, then verify every reviewed candidate's identity."""
import sqlite3
import shutil
import numpy as np
import pandas as pd
from common import *

def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    receipt = PRIVATE / 'freeze.json'
    if receipt.exists():
        old = read(receipt)
        assert sha(PRIVATE / 'reviews.sqlite3') == old['snapshot_sha256']
        assert sha(PRIVATE / 'reviewed_candidates.parquet') == old['joined_sha256']
        print('Verified existing frozen snapshot:', old['created_at'])
        return
    dbpath = PRIVATE / 'reviews.sqlite3'
    assert not dbpath.exists(), 'A partial freeze must be inspected before retrying.'
    with sqlite3.connect(f'file:{CAMPAIGN}/reviews.sqlite3?mode=ro', uri=True) as src:
        with sqlite3.connect(dbpath) as dst:
            src.backup(dst)
    with sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True) as db:
        labels = pd.read_sql_query('SELECT * FROM reviews', db)
        cohorts = pd.read_sql_query('SELECT * FROM cohorts', db).set_index('scene')
        history_count = db.execute('SELECT COUNT(*) FROM history').fetchone()[0]
    assert len(labels) == 1120 and labels.candidate_id.is_unique
    assert sorted(labels.scene.unique()) == SCENES
    assert labels.label.isin(['collision', 'non_collision', 'ambiguous']).all()
    assert labels.groupby('scene').size().eq(80).all()
    assert cohorts.model_sha256.eq(MODEL_SHA).all()
    joined, sources = [], []
    for scene in SCENES:
        root = CAMPAIGN / f'scenes/scene{scene:04d}'
        for stage in ['background', 'minip', 'measurement', 'tracking', 'inference', 'assets']:
            marker = read(root / f'stages/{stage}.json')
            assert marker['status'] == 'PASS' and marker['smoke_frames'] == 0
        cohort = cohorts.loc[scene]
        manifest_path = root / 'review/manifest.json'
        manifest = read(manifest_path)
        assert sha(manifest_path) == cohort.manifest_sha256
        assert manifest['model_sha256'] == MODEL_SHA
        assert manifest['run_id'] == cohort.run_id
        assert sha(root / 'inference/predictions.parquet') == cohort.prediction_sha256 == manifest['prediction_sha256']
        assert sha(root / 'inference/top80.parquet') == manifest['top80_sha256']
        top = pd.read_parquet(root / 'inference/top80.parquet')
        assert top.model_sha256.eq(MODEL_SHA).all()
        assert top.prediction_sha256.eq(cohort.prediction_sha256).all()
        assert top.review_rank.tolist() == list(range(1, 81))
        assert (np.diff(top.score) <= 1e-12).all()
        candidates = pd.DataFrame(manifest['candidates'])
        assert candidates.candidate_id.tolist() == top.candidate_id.tolist()
        assert candidates.review_rank.tolist() == top.review_rank.tolist()
        selected = labels.loc[labels.scene.eq(scene)]
        selected = selected.merge(candidates[['candidate_id','asset_sha256']], on='candidate_id', validate='one_to_one', suffixes=('', '_manifest'))
        assert len(selected) == 80 and selected.asset_sha256.eq(selected.asset_sha256_manifest).all()
        merged = top.merge(selected[['candidate_id','label','note','updated_at','asset_sha256']], on='candidate_id', validate='one_to_one')
        assert len(merged) == 80
        merged['status'] = merged.label
        merged['review_score_rank'] = merged.review_rank
        merged['collision_score'] = merged.score
        merged['review_stratum'] = 'top_score'
        merged['review_inclusion_probability'] = 1.
        merged['binary_label'] = merged.label.map({'collision':1, 'non_collision':0})
        joined.append(merged)
        tracking_marker = read(root / 'stages/tracking.json')['result']
        for relative, expected in [('tracking/smoothed.parquet', tracking_marker['outputs']['smoothed.parquet']),
                                   ('measurement/particles_3d.csv', read(root / 'stages/measurement.json')['result']['particles_sha256'])]:
            actual = sha(root / relative)
            assert actual == expected, (scene, relative)
            sources.append({'scene': scene, 'path': str((root / relative).relative_to(WORK)), 'sha256': actual})
        for relative in ['review/manifest.json','inference/top80.parquet','input/acquisition.yaml','stages/inference.json','stages/measurement.json','stages/tracking.json']:
            target = PRIVATE / f'scene{scene:04d}' / Path(relative).name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(root / relative, target)
        print(f'Frozen scene {scene}: 80 labels, manifest/model/prediction/rank/source hashes verified', flush=True)
    all_rows = pd.concat(joined, ignore_index=True).sort_values(['scene_num','review_rank'])
    all_rows.to_parquet(PRIVATE / 'reviewed_candidates.parquet', index=False)
    labels.to_csv(PRIVATE / 'review_annotations.csv', index=False)
    save(receipt, {'status':'PASS','created_at':stamp(), 'snapshot_sha256':sha(dbpath),
        'joined_sha256':sha(PRIVATE / 'reviewed_candidates.parquet'), 'model_sha256':MODEL_SHA,
        'review_count':len(labels),'history_count':history_count,'counts':labels.label.value_counts().to_dict(),
        'last_review_at':labels.updated_at.max(),'sources':sources,
        'scope':'Snapshot is private. Live review DB, original images and frozen scientific pipeline remain unchanged.'})

if __name__ == '__main__':
    main()
