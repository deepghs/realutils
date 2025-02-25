"""
Face feature extraction and comparison module.

This module provides functionality for extracting facial features and comparing faces using embedding vectors.
It includes methods for face normalization, feature extraction, and similarity calculation between faces.
The module uses pre-trained models from Hugging Face Hub for face feature extraction.
"""

import json
from typing import Union, List, Optional

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from imgutils.data import ImageTyping, load_image
from imgutils.utils import ts_lru_cache, open_onnx_model
from skimage.transform import SimilarityTransform

from .base import _REPO_ID, Face, _DEFAULT_MODEL
from .detect import _open_ref_info


@ts_lru_cache()
def _open_extract_model(model_name: str):
    """
    Open and cache the face feature extraction model.

    :param model_name: Name of the model to load
    :type model_name: str
    :return: Tuple containing model session, input name, input shape and output names
    :rtype: tuple
    """
    model_filename = _open_ref_info(model_name)['extract']
    session = open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/{model_filename}'
    ))
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    input_shape = tuple(input_cfg.shape[2:4][::-1])
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, input_shape, output_names


@ts_lru_cache()
def _get_default_threshold(model_name: str) -> float:
    """
    Get the default similarity threshold for face comparison.

    :param model_name: Name of the model
    :type model_name: str
    :return: Optimal threshold value
    :rtype: float
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/metrics.json',
    )) as f:
        return json.load(f)['optimal_threshold']


_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32
)


def estimate_norm(lmk: np.ndarray, image_size: int = 112):
    """
    Estimate normalization parameters for face alignment.

    :param lmk: 5x2 array of facial landmarks
    :type lmk: np.ndarray
    :param image_size: Target image size, must be 112
    :type image_size: int
    :return: Transformation parameters
    :rtype: np.ndarray
    :raises AssertionError: If landmark shape is invalid or image_size is not 112
    """
    assert lmk.shape == (5, 2), 'Keypoints should be 5x2.'
    assert image_size == 112, 'Image size should be 112.'
    ratio = float(image_size) / 112.0
    diff_x = 0
    dst = _ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = SimilarityTransform()
    tform.estimate(lmk, dst)
    mparams = tform.params[0:2, :]
    return mparams


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112):
    """
    Normalize and crop face image according to landmarks.

    :param img: Input image
    :type img: np.ndarray
    :param landmark: 5x2 array of facial landmarks
    :type landmark: np.ndarray
    :param image_size: Target image size
    :type image_size: int
    :return: Normalized face image
    :rtype: np.ndarray
    """
    mparams = estimate_norm(landmark, image_size)
    # noinspection PyTypeChecker
    warped = cv2.warpAffine(img, mparams, (image_size, image_size), borderValue=0.0)
    return warped


def isf_extract_face(image: ImageTyping, face: Face, model_name: str = _DEFAULT_MODEL, no_write: bool = False):
    """
    Extract face embedding features from an image.

    :param image: Input image
    :type image: ImageTyping
    :param face: Face object containing keypoints
    :type face: Face
    :param model_name: Name of the model to use
    :type model_name: str
    :param no_write: If True, don't write embedding to face object
    :type no_write: bool
    :return: Face embedding vector
    :rtype: np.ndarray
    """
    image = load_image(image, force_background='white', mode='RGB')

    session, input_name, input_shape, output_names = _open_extract_model(model_name=model_name)
    assert input_shape[0] == input_shape[1], f'Input shape is not a square - {input_shape!r}.'
    input_size = input_shape[0]
    # noinspection PyTypeChecker
    blob = norm_crop(np.array(image), np.array(face.keypoints), input_size)
    blob = blob.astype(np.float32)
    blob = (blob - 127.5) / 127.5
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # Change to CHW

    embedding = session.run(output_names, {input_name: blob})[0][0]
    if isinstance(face, Face) and not no_write:
        face.embedding = embedding
    return embedding


def isf_face_batch_similarity(embs: Union[List[np.ndarray], np.ndarray]):
    """
    Calculate similarity matrix for a batch of face embeddings.

    :param embs: List or array of face embeddings
    :type embs: Union[List[np.ndarray], np.ndarray]
    :return: Similarity matrix
    :rtype: np.ndarray
    """
    if isinstance(embs, (list, tuple)):
        embs = np.stack(embs)

    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
    return embs @ embs.T


def isf_face_similarity(emb1: np.ndarray, emb2: np.ndarray):
    """
    Calculate similarity between two face embeddings.

    :param emb1: First face embedding
    :type emb1: np.ndarray
    :param emb2: Second face embedding
    :type emb2: np.ndarray
    :return: Similarity score
    :rtype: float
    """
    return isf_face_batch_similarity([emb1, emb2])[0, 1].item()


def isf_face_batch_same(embs: Union[List[np.ndarray], np.ndarray], model_name: str = _DEFAULT_MODEL,
                        threshold: Optional[float] = None):
    """
    Determine if faces in a batch are the same person using similarity threshold.

    :param embs: List or array of face embeddings
    :type embs: Union[List[np.ndarray], np.ndarray]
    :param model_name: Name of the model to use
    :type model_name: str
    :param threshold: Similarity threshold, if None uses default
    :type threshold: Optional[float]
    :return: Boolean matrix indicating matching faces
    :rtype: np.ndarray
    """
    if threshold is None:
        threshold = _get_default_threshold(model_name=model_name)
    return isf_face_batch_similarity(embs) >= threshold


def isf_face_same(emb1: np.ndarray, emb2: np.ndarray, model_name: str = _DEFAULT_MODEL,
                  threshold: Optional[float] = None) -> float:
    """
    Determine if two faces are the same person.

    :param emb1: First face embedding
    :type emb1: np.ndarray
    :param emb2: Second face embedding
    :type emb2: np.ndarray
    :param model_name: Name of the model to use
    :type model_name: str
    :param threshold: Similarity threshold, if None uses default
    :type threshold: Optional[float]
    :return: Boolean indicating if faces match
    :rtype: float
    """
    return isf_face_batch_same([emb1, emb2], model_name=model_name, threshold=threshold)[0, 1].item()
