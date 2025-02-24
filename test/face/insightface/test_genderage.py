import pytest
from hbutils.testing import tmatrix

from realutils.face.insightface import Face, isf_genderage
from realutils.face.insightface.genderage import _open_attribute_model
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_attribute_model.cache_clear()


@pytest.fixture()
def ins_file_multiple():
    return get_testfile("yolo", 'multiple.jpg')


@pytest.fixture()
def ga_file_multiple():
    return [('F', 31), ('M', 61), ('F', 25), ('M', 22), ('F', 26)]


@pytest.fixture()
def faces_file_multiple():
    return [Face(bbox=(1076.8504638671875,
                       729.2430419921875,
                       1259.576904296875,
                       987.3271484375),
                 det_score=0.929023265838623,
                 keypoints=[(1129.4158935546875, 830.8023681640625),
                            (1217.480224609375, 831.9884643554688),
                            (1173.658935546875, 886.8854370117188),
                            (1121.6807861328125, 910.5137329101562),
                            (1213.4404296875, 911.7223510742188)], ),
            Face(bbox=(900.3935546875,
                       312.42401123046875,
                       1067.79931640625,
                       521.516845703125),
                 det_score=0.9024127721786499,
                 keypoints=[(967.7745971679688, 386.7110595703125),
                            (1037.3992919921875, 408.15423583984375),
                            (1002.6278076171875, 440.20550537109375),
                            (950.5680541992188, 458.22845458984375),
                            (1012.9031372070312, 476.1373596191406)], ),
            Face(bbox=(1381.470458984375,
                       532.9207153320312,
                       1542.2178955078125,
                       715.8306884765625),
                 det_score=0.8893781900405884,
                 keypoints=[(1409.1314697265625, 605.0761108398438),
                            (1478.549072265625, 605.6742553710938),
                            (1434.15576171875, 652.44970703125),
                            (1420.3602294921875, 671.822021484375),
                            (1479.8131103515625, 672.0046997070312)], ),
            Face(bbox=(556.5820922851562,
                       293.680908203125,
                       758.1030883789062,
                       553.5236206054688),
                 det_score=0.8887537121772766,
                 keypoints=[(646.1146240234375, 387.3534240722656),
                            (729.4882202148438, 390.5887145996094),
                            (704.3795776367188, 446.2774963378906),
                            (658.779296875, 492.882568359375),
                            (719.3273315429688, 494.32952880859375)], ),
            Face(bbox=(1198.9088134765625,
                       216.38607788085938,
                       1328.093505859375,
                       403.63507080078125),
                 det_score=0.8424620628356934,
                 keypoints=[(1230.3328857421875, 283.1265563964844),
                            (1286.682861328125, 300.26031494140625),
                            (1242.763671875, 328.5237731933594),
                            (1218.411865234375, 346.5495300292969),
                            (1269.6983642578125, 359.8655700683594)], )]


@pytest.fixture()
def ins_file_2girls():
    return get_testfile("yolo", '2girls.jpg')


@pytest.fixture()
def ga_file_2girls():
    return [('F', 23), ('F', 26)]


@pytest.fixture()
def faces_file_2girls():
    return [Face(bbox=(724.8386840820312,
                       151.70326232910156,
                       1124.8883056640625,
                       731.4854125976562),
                 det_score=0.8008737564086914,
                 keypoints=[(786.199462890625, 411.8285217285156),
                            (969.9263916015625, 414.9202575683594),
                            (846.927001953125, 543.0126953125),
                            (804.715576171875, 602.2054443359375),
                            (960.6113891601562, 606.3782348632812)], ),
            Face(bbox=(156.7660369873047,
                       263.2521667480469,
                       516.4180908203125,
                       718.8055419921875),
                 det_score=0.7389086484909058,
                 keypoints=[(224.55078125, 465.1809387207031),
                            (400.0935974121094, 457.0533752441406),
                            (304.2454833984375, 580.7157592773438),
                            (260.9578552246094, 623.0034790039062),
                            (398.64520263671875, 616.4241943359375)], )]


@pytest.fixture()
def ins_file_solo():
    return get_testfile("yolo", 'solo.jpg')


@pytest.fixture()
def ga_file_solo():
    return [('F', 25)]


@pytest.fixture()
def faces_file_solo():
    return [Face(bbox=(165.1768341064453,
                       84.70780181884766,
                       248.22479248046875,
                       198.0118408203125),
                 det_score=0.8186577558517456,
                 keypoints=[(184.73941040039062, 122.77735900878906),
                            (216.00198364257812, 139.8385009765625),
                            (181.3822784423828, 150.35955810546875),
                            (172.00160217285156, 164.23977661132812),
                            (195.6698455810547, 178.4802703857422)], )]


@pytest.fixture()
def ins_file_3_cosplay():
    return get_testfile("yolo", '3+cosplay.jpg')


@pytest.fixture()
def ga_file_3_cosplay():
    return [('F', 32), ('F', 25), ('F', 29)]


@pytest.fixture()
def faces_file_3_cosplay():
    return [Face(bbox=(353.07904052734375,
                       227.432373046875,
                       409.73779296875,
                       301.43707275390625),
                 det_score=0.8783131837844849,
                 keypoints=[(373.4512634277344, 256.00341796875),
                            (398.38116455078125, 258.5744934082031),
                            (386.3522033691406, 273.4198913574219),
                            (372.40289306640625, 284.2008361816406),
                            (391.31072998046875, 286.3553161621094)], ),
            Face(bbox=(384.099609375,
                       61.932193756103516,
                       427.1168212890625,
                       115.54805755615234),
                 det_score=0.8206616044044495,
                 keypoints=[(406.5011901855469, 78.86163330078125),
                            (422.58184814453125, 83.0195541381836),
                            (418.5873107910156, 89.95550537109375),
                            (406.0752258300781, 100.5828857421875),
                            (417.7054443359375, 103.88252258300781)], ),
            Face(bbox=(195.70213317871094,
                       106.00887298583984,
                       242.3323974609375,
                       160.3513946533203),
                 det_score=0.802963376045227,
                 keypoints=[(203.7552032470703, 131.0619354248047),
                            (221.5390167236328, 122.52863311767578),
                            (214.71810913085938, 138.91998291015625),
                            (214.9202880859375, 149.88510131835938),
                            (228.81692504882812, 142.75189208984375)], )]


@pytest.fixture()
def ga_tuples(ins_file_multiple, ga_file_multiple, faces_file_multiple, ins_file_2girls, ga_file_2girls,
              faces_file_2girls, ins_file_solo, ga_file_solo, faces_file_solo, ins_file_3_cosplay, ga_file_3_cosplay,
              faces_file_3_cosplay):
    return {
        'multiple': (ins_file_multiple, ga_file_multiple, faces_file_multiple),
        '2girls': (ins_file_2girls, ga_file_2girls, faces_file_2girls),
        'solo': (ins_file_solo, ga_file_solo, faces_file_solo),
        '3_cosplay': (ins_file_3_cosplay, ga_file_3_cosplay, faces_file_3_cosplay),
    }


@pytest.mark.unittest
class TestFaceInsightfaceGenderage:
    @pytest.mark.parametrize(*tmatrix({
        'sample_name': ['multiple', '2girls', 'solo', '3_cosplay'],
        'model_name': ['buffalo_s', 'buffalo_l'],
    }))
    def test_isf_genderage(self, sample_name, ga_tuples, model_name):
        image_file, expected_result, faces = ga_tuples[sample_name]
        assert expected_result == [isf_genderage(image_file, face, model_name=model_name) for face in faces]
        assert expected_result == [(face.gender, face.age) for face in faces]

    @pytest.mark.parametrize(*tmatrix({
        'sample_name': ['multiple', '2girls', 'solo', '3_cosplay'],
        'model_name': ['buffalo_s', 'buffalo_l'],
    }))
    def test_isf_genderage_no_write(self, sample_name, ga_tuples, model_name):
        image_file, expected_result, faces = ga_tuples[sample_name]
        assert (expected_result ==
                [isf_genderage(image_file, face, model_name=model_name, no_write=True) for face in faces])
        assert all(face.gender is None for face in faces)
        assert all(face.age is None for face in faces)

    @pytest.mark.parametrize(*tmatrix({
        'sample_name': ['multiple', '2girls', 'solo', '3_cosplay'],
        'model_name': ['buffalo_s', 'buffalo_l'],
    }))
    def test_isf_genderage_bbox(self, sample_name, ga_tuples, model_name):
        image_file, expected_result, faces = ga_tuples[sample_name]
        assert expected_result == [isf_genderage(image_file, face.bbox, model_name=model_name) for face in faces]
        assert all(face.gender is None for face in faces)
        assert all(face.age is None for face in faces)
