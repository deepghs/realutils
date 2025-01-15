"""
Overview:
    Detect objects in generic images, using the official pretrained models.

    Models are hosted on `deepghs/yolos <https://huggingface.co/deepghs/yolos>`_.

    .. image:: yolo_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the censor detect models:

    .. image:: yolo_benchmark.plot.py.svg
        :align: center

"""
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.generic import yolo_predict

_REPO_ID = 'deepghs/yolos'
_DEFAULT_MODEL = 'yolo11s'


def detect_by_yolo(image: ImageTyping, model_name: str = _DEFAULT_MODEL,
                   conf_threshold: float = 0.3, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect object in generic images.

    :param image: The input image to be analyzed. Can be a file path, URL, or image data.
    :type image: ImageTyping

    :param model_name: Optional custom model name. Default is `yolo11s`.
    :type model_name: str

    :param conf_threshold: The confidence threshold for detections. Only detections with
                           confidence above this value will be returned. Default is 0.3.
    :type conf_threshold: float

    :param iou_threshold: The Intersection over Union (IoU) threshold for non-maximum
                          suppression. Detections with IoU above this value will be merged.
                          Default is 0.7.
    :type iou_threshold: float

    :return: A list of tuples, each containing:
        - A tuple of four integers (x0, y0, x1, y1) representing the bounding box
        - A string indicating the type of detection (e.g. 'person', 'cat', etc)
        - A float representing the confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :raises ValueError: If an invalid level is provided.
    :raises RuntimeError: If the model fails to load or process the image.
    """
    return yolo_predict(
        image=image,
        repo_id=_REPO_ID,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
