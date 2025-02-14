"""
Overview:
    Detect human heads in both real photo and anime images.

    Trained with `deepghs/anime_head_detection <https://huggingface.co/datasets/deepghs/anime_head_detection>`_ \
    and open-sourced real photos datasets.

    .. image:: head_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the head detect models:

    .. image:: head_detect_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/real_head_detection <https://huggingface.co/deepghs/real_head_detection>`_.

"""
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.generic import yolo_predict

_REPO_ID = 'deepghs/real_head_detection'


def detect_heads(image: ImageTyping, model_name: str = 'head_detect_v0_s_yv11',
                 conf_threshold: float = 0.2, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human heads in both real photo and anime images using YOLO models.

    This function applies a pre-trained YOLO model to detect heads in the given anime image.
    It supports different model levels and versions, allowing users to balance between
    detection speed and accuracy.

    :param image: The input image for head detection. Can be various image types supported by ImageTyping.
    :type image: ImageTyping

    :param model_name: Optional custom model name. If provided, it overrides the auto-generated model name.
    :type model_name: str

    :param conf_threshold: The confidence threshold for detections. Only detections with confidence
                           scores above this threshold will be returned. Default is 0.2.
    :type conf_threshold: float

    :param iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.
                          Detections with IoU above this threshold will be merged. Default is 0.7.
    :type iou_threshold: float

    :return: A list of detected heads. Each head is represented by a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - The string 'head' (as this function only detects heads)
             - The confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
