"""
Overview:
    Detect human faces in both real photo and anime images.

    Trained with `deepghs/anime_face_detection <https://huggingface.co/datasets/deepghs/anime_face_detection>`_ \
    and open-sourced real photos datasets.

    .. image:: face_detect_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the face detect models:

    .. image:: face_detect_benchmark.plot.py.svg
        :align: center

    The models are hosted on
    `huggingface - deepghs/real_face_detection <https://huggingface.co/deepghs/real_face_detection>`_.

"""
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.generic import yolo_predict

_REPO_ID = 'deepghs/real_face_detection'


def detect_faces(image: ImageTyping, model_name: str = 'face_detect_v0_s_yv11',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7, **kwargs) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect human faces in both real photo and anime images using YOLO models.

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
        >>> from realutils.detect import detect_faces
        >>>
        >>> detect_faces('yolo/solo.jpg')
        [((157, 94, 252, 208), 'face', 0.8836570382118225)]
        >>> detect_faces('yolo/2girls.jpg')
        [((718, 154, 1110, 728), 'face', 0.8841166496276855), ((157, 275, 519, 715), 'face', 0.8668240904808044)]
        >>> detect_faces('yolo/3+cosplay.jpg')
        [((349, 227, 413, 305), 'face', 0.8543888330459595), ((383, 61, 432, 117), 'face', 0.8080574870109558), ((194, 107, 245, 162), 'face', 0.8035706877708435)]
        >>> detect_faces('yolo/multiple.jpg')
        [((1070, 728, 1259, 985), 'face', 0.8765808939933777), ((548, 286, 760, 558), 'face', 0.8693087697029114), ((896, 315, 1067, 520), 'face', 0.8671919107437134), ((1198, 220, 1342, 406), 'face', 0.8485829830169678), ((1376, 526, 1546, 719), 'face', 0.8469308018684387)]

        >>> from imgutils.detect import detection_visualize
        >>> from matplotlib import pyplot as plt
        >>>
        >>> image = 'yolo/solo.jpg'
        >>> result = detect_faces(image)
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
