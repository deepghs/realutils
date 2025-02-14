import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect.head import detect_heads
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectHead:
    def test_detect_heads_solo(self):
        detection = detect_heads(get_testfile('yolo', 'solo.jpg'))
        similarity = detection_similarity(detection, [
            ((162, 47, 305, 210), 'head', 0.7701659202575684)
        ])
        assert similarity >= 0.9

    def test_detect_heads_2girls(self):
        detection = detect_heads(get_testfile('yolo', '2girls.jpg'))
        similarity = detection_similarity(detection, [
            ((683, 48, 1199, 754), 'head', 0.8410779237747192),
            ((105, 91, 570, 734), 'head', 0.8339194059371948)
        ])
        assert similarity >= 0.9

    def test_detect_heads_3cosplays(self):
        detection = detect_heads(get_testfile('yolo', '3+cosplay.jpg'))
        similarity = detection_similarity(detection, [
            ((329, 194, 426, 309), 'head', 0.8123012781143188),
            ((359, 20, 448, 122), 'head', 0.8047150373458862),
            ((185, 81, 265, 166), 'head', 0.7797152996063232)
        ])
        assert similarity >= 0.9

    def test_detect_heads_multiple(self):
        detection = detect_heads(get_testfile('yolo', 'multiple.jpg'))
        similarity = detection_similarity(detection, [
            ((867, 259, 1084, 527), 'head', 0.8264595866203308),
            ((1364, 448, 1583, 724), 'head', 0.8254891633987427),
            ((480, 201, 781, 565), 'head', 0.8191508054733276),
            ((1189, 175, 1398, 412), 'head', 0.8097156286239624),
            ((1028, 671, 1277, 992), 'head', 0.8084591627120972)
        ])
        assert similarity >= 0.9
