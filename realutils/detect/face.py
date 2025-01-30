"""
Overview:
    Detect human faces in real images.

    Inspired by project `akanametov/yolo-face <https://github.com/akanametov/yolo-face>`_.

    .. image:: face_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the face detect models:

    .. image:: face_detect_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/yolo-face <https://huggingface.co/deepghs/yolo-face>`_.

"""
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.generic import yolo_predict

_REPO_ID = 'deepghs/yolo-face'


def detect_real_faces(image: ImageTyping, model_name: str = 'yolov11s-face',
                      conf_threshold: float = 0.25, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human faces in real images using YOLO models.

    This function applies a pre-trained YOLO model to detect faces in the given anime image.
    It supports different model levels and versions, allowing users to balance between
    detection speed and accuracy.

    :param image: The input image for face detection. Can be various image types supported by ImageTyping.
    :type image: ImageTyping

    :param model_name: Optional custom model name. If provided, it overrides the auto-generated model name.
    :type model_name: str

    :param conf_threshold: The confidence threshold for detections. Only detections with confidence
                           scores above this threshold will be returned. Default is 0.25.
    :type conf_threshold: float

    :param iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.
                          Detections with IoU above this threshold will be merged. Default is 0.7.
    :type iou_threshold: float

    :return: A list of detected faces. Each face is represented by a tuple containing:
             - Bounding box coordinates as (x0, y0, x1, y1)
             - The string 'face' (as this function only detects faces)
             - The confidence score of the detection
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    Examples::
        >>> from imgutils.detect import detect_faces, detection_visualize
        >>>
        >>> image = 'mostima_post.jpg'
        >>> result = detect_faces(image)  # detect it
        >>> result
        [
            ((29, 441, 204, 584), 'face', 0.7874319553375244),
            ((346, 59, 529, 275), 'face', 0.7510495185852051),
            ((606, 51, 895, 336), 'face', 0.6986488103866577)
        ]
        >>>
        >>> # visualize it
        >>> from matplotlib import pyplot as plt
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
