import pytest
from imgutils.detect import detection_similarity
from imgutils.generic.yolo import _open_models_for_repo_id

from realutils.detect.person import detect_persons
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id.cache_clear()


@pytest.mark.unittest
class TestDetectPerson:
    def test_detect_persons_solo(self):
        detection = detect_persons(get_testfile('yolo', 'solo.jpg'))
        similarity = detection_similarity(detection, [
            ((0, 30, 398, 599), 'person', 0.926707923412323)
        ])
        assert similarity >= 0.9

    def test_detect_persons_2girls(self):
        detection = detect_persons(get_testfile('yolo', '2girls.jpg'))
        similarity = detection_similarity(detection, [
            ((0, 74, 760, 1598), 'person', 0.7578195333480835),
            ((437, 33, 1200, 1600), 'person', 0.6875205039978027)
        ])
        assert similarity >= 0.9

    def test_detect_persons_3cosplays(self):
        detection = detect_persons(get_testfile('yolo', '3+cosplay.jpg'))
        similarity = detection_similarity(detection, [
            ((106, 69, 347, 591), 'person', 0.8794167041778564),
            ((326, 14, 592, 534), 'person', 0.8018194437026978),
            ((167, 195, 676, 675), 'person', 0.5351650714874268)
        ])
        assert similarity >= 0.9

    def test_detect_persons_multiple(self):
        detection = detect_persons(get_testfile('yolo', 'multiple.jpg'))
        similarity = detection_similarity(detection, [
            ((1305, 441, 1891, 1534), 'person', 0.8789498805999756),
            ((206, 191, 932, 1533), 'person', 0.8423126935958862),
            ((1054, 170, 1417, 1055), 'person', 0.8138357996940613),
            ((697, 659, 1473, 1534), 'person', 0.7926754951477051),
            ((685, 247, 1128, 1526), 'person', 0.5261526703834534),
            ((690, 251, 1125, 1126), 'person', 0.4193646311759949)
        ])
        assert similarity >= 0.9
