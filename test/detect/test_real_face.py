import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect.real_face import detect_real_faces
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectRealFace:
    def test_detect_real_faces_solo(self):
        detection = detect_real_faces(get_testfile('yolo', 'solo.jpg'))
        similarity = detection_similarity(detection, [
            ((168, 79, 245, 199), 'face', 0.7996422052383423),
        ])
        assert similarity >= 0.9

    def test_detect_real_faces_2girls(self):
        detection = detect_real_faces(get_testfile('yolo', '2girls.jpg'))
        similarity = detection_similarity(detection, [
            ((721, 152, 1082, 726), 'face', 0.8811314702033997),
            ((158, 263, 509, 714), 'face', 0.8745490908622742),
        ])
        assert similarity >= 0.9

    def test_detect_real_faces_3cosplays(self):
        detection = detect_real_faces(get_testfile('yolo', '3+cosplay.jpg'))
        similarity = detection_similarity(detection, [
            ((351, 228, 410, 302), 'face', 0.8392542600631714),
            ((384, 63, 427, 116), 'face', 0.8173024654388428),
            ((195, 109, 246, 161), 'face', 0.8126493692398071)
        ])
        assert similarity >= 0.9

    def test_detect_real_faces_multiple(self):
        detection = detect_real_faces(get_testfile('yolo', 'multiple.jpg'))
        similarity = detection_similarity(detection, [
            ((1074, 732, 1258, 987), 'face', 0.8792377710342407),
            ((1378, 536, 1541, 716), 'face', 0.8607611656188965),
            ((554, 295, 759, 557), 'face', 0.8541485071182251),
            ((897, 315, 1068, 520), 'face', 0.8539882898330688),
            ((1194, 230, 1329, 403), 'face', 0.8324605226516724)
        ])
        assert similarity >= 0.9
