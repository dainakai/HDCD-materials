"""Guard the three-method comparison against label leakage and clip misindexing."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from threadpoolctl import threadpool_limits

from ra3d import minip

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('directory', ['baseline2d', 'minip2d'])
def test_minip_prediction_is_independent_of_truth_labels(directory):
    frame = pd.read_parquet(ROOT / 'data' / directory / 'validation/seed260745/full.parquet').head(256)
    bundle = minip.load(ROOT / 'data/models' / directory)
    unlabelled = frame.drop(columns=minip.LABEL_COLUMNS, errors='ignore')
    changed = unlabelled.copy()
    for index, column in enumerate(minip.LABEL_COLUMNS):
        changed[column] = index + 100
    with threadpool_limits(limits=2):
        expected = minip.predict(unlabelled, bundle, 260745)
        actual = minip.predict(changed, bundle, 260745)
    pd.testing.assert_frame_equal(actual, expected)


def test_minip_reader_preserves_clip_frame_and_precision(tmp_path):
    Image.fromarray(np.full((1024, 1024), 127, dtype=np.uint8)).save(tmp_path / '000049.png')
    for method, dtype in [('depth-free', np.float32), ('minip-2d', np.float64)]:
        reader = minip.MinIPReader(tmp_path, method)
        assert reader.acquisition.frames.index.tolist() == [48]
        assert reader.intensity(48, 0).dtype == dtype
    Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8)).save(tmp_path / '000000.png')
    with pytest.raises(ValueError, match='one-based'):
        minip.MinIPReader(tmp_path, 'minip-2d')

