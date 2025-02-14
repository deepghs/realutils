import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect.face import detect_faces
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectFace:
    def test_detect_faces_solo(self):
        detection = detect_faces(get_testfile('yolo', 'solo.jpg'))
        similarity = detection_similarity(detection, [
            ((157, 94, 252, 208), 'face', 0.8836570382118225)
        ])
        assert similarity >= 0.9

    def test_detect_faces_2girls(self):
        detection = detect_faces(get_testfile('yolo', '2girls.jpg'))
        similarity = detection_similarity(detection, [
            ((718, 154, 1110, 728), 'face', 0.8841166496276855),
            ((157, 275, 519, 715), 'face', 0.8668240904808044)
        ])
        assert similarity >= 0.9

    def test_detect_faces_3cosplays(self):
        detection = detect_faces(get_testfile('yolo', '3+cosplay.jpg'))
        similarity = detection_similarity(detection, [
            ((349, 227, 413, 305), 'face', 0.8543888330459595),
            ((383, 61, 432, 117), 'face', 0.8080574870109558),
            ((194, 107, 245, 162), 'face', 0.8035706877708435)
        ])
        assert similarity >= 0.9

    def test_detect_faces_multiple(self):
        detection = detect_faces(get_testfile('yolo', 'multiple.jpg'))
        similarity = detection_similarity(detection, [
            ((1070, 728, 1259, 985), 'face', 0.8765808939933777),
            ((548, 286, 760, 558), 'face', 0.8693087697029114),
            ((896, 315, 1067, 520), 'face', 0.8671919107437134),
            ((1198, 220, 1342, 406), 'face', 0.8485829830169678),
            ((1376, 526, 1546, 719), 'face', 0.8469308018684387)
        ])
        assert similarity >= 0.9
