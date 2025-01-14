import os.path

import numpy as np
import pytest

from realutils.metrics import get_dinov2_embedding
from realutils.metrics.dinov2 import _get_dinov2_model, _get_preprocess_config
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model():
    try:
        yield
    finally:
        _get_dinov2_model.cache_clear()
        _get_preprocess_config.cache_clear()


@pytest.mark.unittest
class TestMetricsDinov2:
    @pytest.mark.parametrize(['filename'], [
        (file,) for file in [
            'unsplash_6JzF8Bf4Uhw.jpg',
            'unsplash_yXAtOOBwFNY.jpg',
            'unsplash_c_s2DQoe4j8.jpg',
            'unsplash_--2IBUMom1I.jpg',
            'unsplash_wIhdsYo9g6w.jpg',
            'unsplash_vYobFdRqxfE.jpg',
            'unsplash_tsStT0oQhY8.jpg',
            'unsplash_ae3CP4sZLV8.jpg',
            'unsplash_-MX1uWzQI8E.jpg',
            'unsplash_iKVGWw6DgRg.jpg'
        ]
    ])
    def test_get_dinov2_embedding(self, filename):
        image_file = get_testfile('dataset', 'unsplash_1000', filename)
        emb_file = get_testfile('dinov2', os.path.splitext(filename)[0] + '.npy')

        embedding = get_dinov2_embedding(image_file)
        np.testing.assert_allclose(embedding, np.load(emb_file), rtol=1e-03, atol=1e-05)
