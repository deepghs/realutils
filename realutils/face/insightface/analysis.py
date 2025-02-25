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
