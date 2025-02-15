"""
Overview:
    Detect persons in both real photo and anime images.

    Trained with `deepghs/anime_person_detection <https://huggingface.co/datasets/deepghs/anime_person_detection>`_ \
    and open-sourced real photos datasets.

    .. image:: person_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the person detect models:

    .. image:: person_detect_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/real_person_detection <https://huggingface.co/deepghs/real_person_detection>`_.

"""
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.generic import yolo_predict

_REPO_ID = 'deepghs/real_person_detection'


def detect_persons(image: ImageTyping, model_name: str = 'person_detect_v0_s_yv11',
                   conf_threshold: float = 0.35, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect persons in both real photo and anime images using YOLO models.

    This function applies a pre-trained YOLO model to detect persons in the given anime image.
    It supports different model levels and versions, allowing users to balance between
    detection speed and accuracy.

    :param image: The input image for person detection. Can be various image types supported by ImageTyping.
    :type image: ImageTyping

    :param model_name: Optional custom model name. If provided, it overrides the auto-generated model name.
    :type model_name: str

    :param conf_threshold: The confidence threshold for detections. Only detections with confidence
                           scores above this threshold will be returned. Default is 0.35.
    :type conf_threshold: float

    :param iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.
                          Detections with IoU above this threshold will be merged. Default is 0.7.
    :type iou_threshold: float

    :return: A list of detected persons. Each person is represented by a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - The string 'person' (as this function only detects persons)
             - The confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :example:
        >>> from realutils.detect import detect_persons
        >>>
        >>> detect_persons('yolo/solo.jpg')
        [((0, 30, 398, 599), 'person', 0.926707923412323)]
        >>> detect_persons('yolo/2girls.jpg')
        [((0, 74, 760, 1598), 'person', 0.7578195333480835), ((437, 33, 1200, 1600), 'person', 0.6875205039978027)]
        >>> detect_persons('yolo/3+cosplay.jpg')
        [((106, 69, 347, 591), 'person', 0.8794167041778564), ((326, 14, 592, 534), 'person', 0.8018194437026978), ((167, 195, 676, 675), 'person', 0.5351650714874268)]
        >>> detect_persons('yolo/multiple.jpg')
        [((1305, 441, 1891, 1534), 'person', 0.8789498805999756), ((206, 191, 932, 1533), 'person', 0.8423126935958862), ((1054, 170, 1417, 1055), 'person', 0.8138357996940613), ((697, 659, 1473, 1534), 'person', 0.7926754951477051), ((685, 247, 1128, 1526), 'person', 0.5261526703834534), ((690, 251, 1125, 1126), 'person', 0.4193646311759949)]

        >>> from imgutils.detect import detection_visualize
        >>> from matplotlib import pyplot as plt
        >>>
        >>> image = 'yolo/solo.jpg'
        >>> result = detect_persons(image)
        >>>
        >>> # visualize it
        >>> plt.imshow(detection_visualize(image, result))
        >>> plt.show()
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
