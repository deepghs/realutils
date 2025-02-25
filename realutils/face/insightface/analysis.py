"""
Face Analysis Module

This module provides functionality for comprehensive face analysis in images, including face detection,
gender/age estimation, and feature extraction. It utilizes pre-trained models from Hugging Face Hub
to perform these tasks.
"""

from typing import Tuple, List

from imgutils.data import ImageTyping
from tqdm import tqdm

from .base import _DEFAULT_MODEL, Face
from .detect import isf_detect_faces
from .extract import isf_extract_face
from .genderage import isf_genderage


def isf_analysis_faces(image: ImageTyping, model_name: str = _DEFAULT_MODEL,
                       input_size: Tuple[int, int] = (640, 640), det_thresh: float = 0.5, nms_thresh: float = 0.4,
                       no_genderage: bool = False, no_extraction: bool = False, silent: bool = False) -> List[Face]:
    """
    Perform comprehensive face analysis on an image, including detection, gender/age estimation, and feature extraction.

    This function processes an image through multiple stages of face analysis:
    1. Face detection to locate faces and their landmarks
    2. Gender and age estimation (optional)
    3. Face feature extraction (optional)

    :param image: Input image for face analysis
    :type image: ImageTyping
    :param model_name: Name of the pre-trained model to use from Hugging Face Hub
    :type model_name: str
    :param input_size: Size to resize input image to before processing (width, height)
    :type input_size: Tuple[int, int]
    :param det_thresh: Detection confidence threshold for face detection
    :type det_thresh: float
    :param nms_thresh: Non-maximum suppression threshold for face detection
    :type nms_thresh: float
    :param no_genderage: If True, skip gender and age estimation
    :type no_genderage: bool
    :param no_extraction: If True, skip face feature extraction
    :type no_extraction: bool
    :param silent: If True, disable progress bar
    :type silent: bool

    :return: List of detected Face objects with analysis results
    :rtype: List[Face]
    """
    faces = isf_detect_faces(
        image=image,
        model_name=model_name,
        input_size=input_size,
        det_thresh=det_thresh,
        nms_thresh=nms_thresh,
    )

    for face in tqdm(faces, disable=silent):
        if not no_genderage:
            isf_genderage(
                image=image,
                face=face,
                model_name=model_name,
            )
        if not no_extraction:
            isf_extract_face(
                image=image,
                face=face,
                model_name=model_name,
            )

    return faces
