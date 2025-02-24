"""
Face detection and analysis module.

The module defines a dataclass Face that encapsulates all face-related attributes
and provides utility methods for working with detection results.

:const _REPO_ID: The default Hugging Face repository ID for the face detection model
:const _DEFAULT_MODEL: The default model name to use for face detection
"""

from dataclasses import dataclass
from typing import Tuple, List, Literal, Optional

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

_REPO_ID = 'deepghs/insightface'
_DEFAULT_MODEL = 'buffalo_l'
_GENDER_NAMES = {'M': 'male', 'F': 'female'}


@dataclass
class Face:
    """
    A dataclass representing detected face information.

    This class stores information about a detected face, including its location,
    detection confidence, facial landmarks, and optional demographic attributes.

    :param bbox: Bounding box coordinates in format (x1, y1, x2, y2)
    :type bbox: Tuple[float, float, float, float]
    :param det_score: Detection confidence score between 0 and 1
    :type det_score: float
    :param keypoints: List of facial keypoint coordinates as (x, y) tuples
    :type keypoints: List[Tuple[float, float]]
    :param gender: Gender classification result, either 'F' for female or 'M' for male
    :type gender: Optional[Literal['F', 'M']]
    :param age: Estimated age in years
    :type age: Optional[int]

    :example:
        >>> face = Face(
        ...     bbox=(100, 200, 300, 400),
        ...     det_score=0.99,
        ...     keypoints=[(150, 250), (200, 250)],
        ...     gender='F',
        ...     age=25
        ... )
    """

    bbox: Tuple[float, float, float, float]
    det_score: float
    keypoints: List[Tuple[float, float]]
    gender: Optional[Literal['F', 'M']] = None
    age: Optional[int] = None

    def to_det_tuple(self) -> Tuple[Tuple[float, float, float, float], str, float]:
        """
        Convert face detection result to a standardized detection tuple format.

        This method formats the face detection information into a tuple that can be
        used with general object detection frameworks or visualization tools.

        :return: A tuple containing (bbox, label, confidence_score)
        :rtype: Tuple[Tuple[float, float, float, float], str, float]

        :example:
            >>> face = Face(bbox=(100, 200, 300, 400), det_score=0.99, keypoints=[])
            >>> bbox, label, score = face.to_det_tuple()
        """
        if self.gender is not None and self.age is not None:
            label = f'{_GENDER_NAMES[self.gender].capitalize()} (Age: {self.age})'
        else:
            label = 'face'
        return self.bbox, label, self.det_score


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = SimilarityTransform(rotation=rot)
    t4 = SimilarityTransform(translation=(output_size / 2,
                                          output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    # noinspection PyTypeChecker
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M
