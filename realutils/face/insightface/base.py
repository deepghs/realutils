"""
Face detection and analysis module.

The module defines a dataclass Face that encapsulates all face-related attributes
and provides utility methods for working with detection results.

:const _REPO_ID: The default Hugging Face repository ID for the face detection model
:const _DEFAULT_MODEL: The default model name to use for face detection
"""

from dataclasses import dataclass
from typing import Tuple, List, Literal, Optional

_REPO_ID = 'deepghs/insightface'
_DEFAULT_MODEL = 'buffalo_l'


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
        return self.bbox, 'face', self.det_score
