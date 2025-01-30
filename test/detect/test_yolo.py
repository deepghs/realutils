import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect import detect_by_yolo
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectYOLO:
    def test_detect_by_yolo_unsplash_aJafJ0sLo6o(self):
        detection = detect_by_yolo(get_testfile('yolo', 'unsplash_aJafJ0sLo6o.jpg'))
        similarity = detection_similarity(detection, [
            ((450, 317, 567, 599), 'person', 0.9004617929458618),
        ])
        assert similarity >= 0.9

    def test_detect_by_yolo_unsplash_n4qQGOBgI7U(self):
        detection = detect_by_yolo(get_testfile('yolo', 'unsplash_n4qQGOBgI7U.jpg'))
        similarity = detection_similarity(detection, [
            ((73, 101, 365, 409), 'vase', 0.9098997116088867),
            ((441, 215, 659, 428), 'vase', 0.622944176197052),
            ((5, 1, 428, 377), 'potted plant', 0.5178268551826477),
        ])
        assert similarity >= 0.9

    def test_detect_by_yolo_unsplash_vUNQaTtZeOo(self):
        detection = detect_by_yolo(get_testfile('yolo', 'unsplash_vUNQaTtZeOo.jpg'))
        similarity = detection_similarity(detection, [
            ((381, 103, 676, 448), 'bird', 0.9061452150344849),
        ])
        assert similarity >= 0.9

    def test_detect_by_yolo_unsplash_YZOqXWF_9pk(self):
        detection = detect_by_yolo(get_testfile('yolo', 'unsplash_YZOqXWF_9pk.jpg'))
        similarity = detection_similarity(detection, [
            ((315, 100, 690, 532), 'horse', 0.9453459978103638),
            ((198, 181, 291, 256), 'horse', 0.917123556137085),
            ((145, 173, 180, 249), 'horse', 0.7972317337989807),
            ((660, 138, 701, 170), 'horse', 0.4843617379665375),
        ])
        assert similarity >= 0.9
