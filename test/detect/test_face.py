from pprint import pprint

import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect.face import detect_real_faces
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
    pass
    # def test_detect_faces_1(self):
    #     detection = detect_real_faces(get_testfile('yolo', 'solo.jpg'))
    #     pprint(detection)
    #     similarity = detection_similarity(detection, [
    #     ])
    #     assert similarity >= 0.9
    #
    # def test_detect_faces_2(self):
    #     detection = detect_real_faces(get_testfile('yolo', '2girls.jpg'))
    #     pprint(detection)
    #     similarity = detection_similarity(detection, [
    #     ])
    #     assert similarity >= 0.9
