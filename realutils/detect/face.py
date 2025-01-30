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

    :example:
        >>> from realutils.detect import detect_real_faces
        >>>
        >>> detect_real_faces('yolo/solo.jpg')
        [((168, 79, 245, 199), 'face', 0.7996422052383423)]
        >>> detect_real_faces('yolo/2girls.jpg')
        [((721, 152, 1082, 726), 'face', 0.8811314702033997), ((158, 263, 509, 714), 'face', 0.8745490908622742)]
        >>> detect_real_faces('yolo/3+cosplay.jpg')
        [((351, 228, 410, 302), 'face', 0.8392542600631714), ((384, 63, 427, 116), 'face', 0.8173024654388428), ((195, 109, 246, 161), 'face', 0.8126493692398071)]
        >>> detect_real_faces('yolo/multiple.jpg')
        [((1074, 732, 1258, 987), 'face', 0.8792377710342407), ((1378, 536, 1541, 716), 'face', 0.8607611656188965), ((554, 295, 759, 557), 'face', 0.8541485071182251), ((897, 315, 1068, 520), 'face', 0.8539882898330688), ((1194, 230, 1329, 403), 'face', 0.8324605226516724)]

        >>> from imgutils.detect import detection_visualize
        >>> from matplotlib import pyplot as plt
        >>>
        >>> image = 'yolo/solo.jpg'
        >>> result = detect_real_faces(image)
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
