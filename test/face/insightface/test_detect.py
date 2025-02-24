import pytest
from imgutils.detect import detection_similarity

from realutils.face.insightface import isf_detect_faces
from test.testings import get_testfile


@pytest.fixture()
def isf_file_multiple():
    return get_testfile("yolo", 'multiple.jpg')


@pytest.fixture()
def det_file_multiple():
    return [((1076.8504638671875, 729.2430419921875, 1259.576904296875, 987.3271484375),
             'face',
             0.929023265838623),
            ((900.3935546875, 312.42401123046875, 1067.79931640625, 521.516845703125),
             'face',
             0.9024127721786499),
            ((1381.470458984375, 532.9207153320312, 1542.2178955078125, 715.8306884765625),
             'face',
             0.8893781900405884),
            ((556.5820922851562, 293.680908203125, 758.1030883789062, 553.5236206054688),
             'face',
             0.8887537121772766),
            ((1198.9088134765625,
              216.38607788085938,
              1328.093505859375,
              403.63507080078125),
             'face',
             0.8424620628356934)]


@pytest.fixture()
def isf_file_2girls():
    return get_testfile("yolo", '2girls.jpg')


@pytest.fixture()
def det_file_2girls():
    return [((724.8386840820312,
              151.70326232910156,
              1124.8883056640625,
              731.4854125976562),
             'face',
             0.8008737564086914),
            ((156.7660369873047, 263.2521667480469, 516.4180908203125, 718.8055419921875),
             'face',
             0.7389086484909058)]


@pytest.fixture()
def isf_file_solo():
    return get_testfile("yolo", 'solo.jpg')


@pytest.fixture()
def det_file_solo():
    return [((165.1768341064453, 84.70780181884766, 248.22479248046875, 198.0118408203125),
             'face',
             0.8186577558517456)]


@pytest.fixture()
def isf_file_3_cosplay():
    return get_testfile("yolo", '3+cosplay.jpg')


@pytest.fixture()
def det_file_3_cosplay():
    return [((353.07904052734375, 227.432373046875, 409.73779296875, 301.43707275390625),
             'face',
             0.8783131837844849),
            ((384.099609375, 61.932193756103516, 427.1168212890625, 115.54805755615234),
             'face',
             0.8206616044044495),
            ((195.70213317871094,
              106.00887298583984,
              242.3323974609375,
              160.3513946533203),
             'face',
             0.802963376045227)]


@pytest.fixture()
def det_tuples(isf_file_multiple, det_file_multiple, isf_file_2girls, det_file_2girls, isf_file_solo, det_file_solo,
               isf_file_3_cosplay, det_file_3_cosplay):
    return {
        'multiple': (isf_file_multiple, det_file_multiple),
        '2girls': (isf_file_2girls, det_file_2girls),
        'solo': (isf_file_solo, det_file_solo),
        '3_cosplay': (isf_file_3_cosplay, det_file_3_cosplay),
    }


@pytest.mark.unittest
class TestFaceInsightfaceDetect:
    @pytest.mark.parametrize(['sample_name'], [
        ('multiple',),
        ('2girls',),
        ('solo',),
        ('3_cosplay',),
    ])
    def test_isf_detect_faces(self, sample_name, det_tuples):
        image_file, expected_detection = det_tuples[sample_name]
        faces = isf_detect_faces(image_file)
        assert detection_similarity(
            [face.to_det_tuple() for face in faces],
            expected_detection,
        ) >= 0.9
