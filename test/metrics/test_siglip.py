import re

import numpy as np
import pytest
from imgutils.generic.siglip import _open_models_for_repo_id

from realutils.metrics.siglip import get_siglip_image_embedding, get_siglip_text_embedding, classify_with_siglip
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestMetricsSiglip:
    @pytest.mark.parametrize(['name'], [
        ('unsplash_sZzmhn2xjQY',),
        ('unsplash_S-8ntPEsSwo',),
        ('unsplash_tB4-ftQ4zyI',),
        ('unsplash_l6KamCXeB4U',),
        ('unsplash__9dAwWA4LD8',),
        ('unsplash_LlsAieNJE70',),
        ('unsplash_HWIOLU7_O6w',),
        ('unsplash_1AAa78W_Ezc',),
        ('unsplash_0TPmrjTXjSs',),
        ('unsplash_0yAVtZiYkJY',)
    ])
    def test_get_siglip_image_embedding(self, name):
        src_image = get_testfile('dataset', 'unsplash_1000', f'{name}.jpg')
        dst_npy = get_testfile('siglip', 'unsplash_1000', f'{name}.npy')
        embedding = get_siglip_image_embedding(src_image)
        expected_embedding = np.load(dst_npy)
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-03, atol=1e-05)

    @pytest.mark.parametrize(['text'], [
        ("a red car parked on the street",),
        ("beautiful sunset over mountain landscape",),
        ("two cats playing with yarn",),
        ("fresh fruits in a wooden bowl",),
        ("person reading book under tree",),
        ("colorful hot air balloon in blue sky",),
        ("children playing soccer in the park",),
        ("rustic cabin surrounded by pine trees",),
        ("waves crashing on sandy beach",),
        ("chef cooking in modern kitchen",),
    ])
    def test_get_siglip_text_embedding(self, text):
        dst_npy = get_testfile('siglip', 'text', re.sub(r'[\W_]+', '_', text).strip('_') + '.npy')
        embedding = get_siglip_text_embedding(text)
        expected_embedding = np.load(dst_npy)
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-03, atol=1e-05)

    def test_classify_with_siglip(self):
        result = classify_with_siglip(
            images=[
                get_testfile('clip_cats.jpg'),
                get_testfile('idolsankaku', '3.jpg'),
            ],
            texts=[
                'a photo of a cat',
                'a photo of 2 cats',
                'a photo of 2 dogs',
                'a photo of a woman',
            ],
        )
        expected_result = np.array(
            [[0.0013782851165160537, 0.27010253071784973, 9.751768811838701e-05, 3.6702780814579228e-09],
             [1.2790776438009743e-08, 4.396981001519862e-09, 3.2838454178119036e-10, 1.0559210750216153e-06]])
        np.testing.assert_allclose(result, expected_result, atol=3e-4)
