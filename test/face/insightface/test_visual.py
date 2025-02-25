import pytest
from hbutils.testing import tmatrix
from imgutils.data import load_image

from realutils.face.insightface import isf_analysis_faces, isf_faces_visualize
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
def isf_file_multiple():
    return get_testfile("yolo", 'multiple.jpg')


@pytest.fixture()
def isf_file_2girls():
    return get_testfile("yolo", '2girls.jpg')


@pytest.fixture()
def isf_file_solo():
    return get_testfile("yolo", 'solo.jpg')


@pytest.fixture()
def isf_file_3_cosplay():
    return get_testfile("yolo", '3+cosplay.jpg')


@pytest.fixture()
def det_tuples(isf_file_multiple, isf_file_2girls, isf_file_solo, isf_file_3_cosplay):
    return {
        'multiple': isf_file_multiple,
        '2girls': isf_file_2girls,
        'solo': isf_file_solo,
        '3_cosplay': isf_file_3_cosplay,
    }


@pytest.mark.unittest
class TestFaceInsightfaceVisual:
    @pytest.mark.parametrize(*tmatrix({
        'sample_name': ['multiple', '2girls', 'solo', '3_cosplay'],
    }))
    def test_isf_faces_visualize(self, sample_name, det_tuples, image_diff):
        image_file = det_tuples[sample_name]
        faces = isf_analysis_faces(image_file, no_extraction=True)
        image = isf_faces_visualize(image_file, faces)
        assert image_diff(
            load_image(get_testfile(f'insightface_visual_{sample_name}.png'), mode='RGB'),
            load_image(image, mode='RGB'),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'sample_name': ['multiple', '2girls', 'solo', '3_cosplay'],
    }))
    def test_isf_faces_visualize_640(self, sample_name, det_tuples, image_diff):
        image_file = det_tuples[sample_name]
        faces = isf_analysis_faces(image_file, no_extraction=True)
        image = isf_faces_visualize(image_file, faces, max_short_edge_size=640)
        assert image_diff(
            load_image(get_testfile(f'insightface_visual_640_{sample_name}.png'), mode='RGB'),
            load_image(image, mode='RGB'),
            throw_exception=False
        ) < 1e-2
