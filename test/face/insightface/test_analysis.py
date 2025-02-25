import glob
import os.path

import numpy as np
import pytest
from hbutils.testing import tmatrix
from natsort import natsorted
from tqdm import tqdm

from realutils.face.insightface import isf_face_batch_similarity, \
    isf_face_batch_same, isf_analysis_faces
from realutils.face.insightface.detect import _open_det_model, _open_ref_info, _get_center_from_cache
from realutils.face.insightface.extract import _open_extract_model
from realutils.face.insightface.genderage import _open_attribute_model
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_ref_info.cache_clear()
        _open_det_model.cache_clear()
        _get_center_from_cache.cache_clear()
        _open_extract_model.cache_clear()
        _open_attribute_model.cache_clear()


@pytest.fixture()
def face_images():
    return natsorted(glob.glob(get_testfile('faces', '*', '*.jpg')))


@pytest.fixture()
def expected_result(face_images):
    image_ch_ids = np.array([int(os.path.basename(os.path.dirname(file))) for file in face_images])
    return image_ch_ids == image_ch_ids[..., None]


@pytest.mark.unittest
class TestFaceInsightfaceAnalysis:
    @pytest.mark.parametrize(*tmatrix({
        'model_name': ['buffalo_s', 'buffalo_l'],
    }))
    def test_isf_analysis_faces_batch_similarities(self, face_images, expected_result, model_name):
        embs = []
        for file in tqdm(face_images):
            face = isf_analysis_faces(file, model_name=model_name)[0]
            embs.append(face.embedding)

        np.testing.assert_allclose(
            isf_face_batch_similarity(embs) >= 0.3,
            expected_result,
        )

    @pytest.mark.parametrize(*tmatrix({
        'model_name': ['buffalo_s', 'buffalo_l'],
    }))
    def test_isf_analysis_faces_batch_same(self, face_images, expected_result, model_name):
        embs = []
        for file in tqdm(face_images):
            face = isf_analysis_faces(file, model_name=model_name)[0]
            embs.append(face.embedding)

        np.testing.assert_allclose(
            isf_face_batch_same(embs),
            expected_result,
        )
