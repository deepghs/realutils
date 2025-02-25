"""
This module provides a comprehensive interface to InsightFace, a 2D/3D face analysis toolkit.
It includes functionalities for face detection, recognition, gender & age estimation, and visualization.

The models are hosted on Hugging Face Hub at `deepghs/insightface <https://huggingface.co/deepghs/insightface>`_,
original project at `deepinsight/insightface <https://github.com/deepinsight/insightface>`_.
"""
from .analysis import isf_analysis_faces
from .base import Face
from .detect import isf_detect_faces
from .extract import isf_extract_face, isf_face_batch_similarity, isf_face_similarity, isf_face_batch_same, \
    isf_face_same
from .genderage import isf_genderage
from .visual import isf_faces_visualize
