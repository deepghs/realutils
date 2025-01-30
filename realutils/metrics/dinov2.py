"""
Overview:
    This module provides functionality for generating embeddings from images using the DINOv2 model.
    It includes utilities for image preprocessing and model inference using ONNX runtime.

    The module supports different DINOv2 model variants and provides configurable preprocessing options.

    The ONNX models are hosted on `deepghs/dinov2_onnx <https://huggingface.co/deepghs/dinov2_onnx>`_.

    This is an overall benchmark of all the dinov2 models:

    .. image:: dinov2_benchmark.plot.py.svg
        :align: center
"""

import json

import numpy as np
from huggingface_hub import hf_hub_download
from imgutils.data import ImageTyping, load_image
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace

_REPO = 'deepghs/dinov2_onnx'
_DEFAULT_MODEL = 'facebook/dinov2-base'


@ts_lru_cache()
def _get_preprocessor(model_name: str):
    """
    Get preprocessing configuration for specified DINOv2 model variant.

    :param model_name: Name of DINOv2 model variant
    :type model_name: str
    :return: Preprocessor for dinov2 model
    :rtype: PillowCompose
    """
    with open(hf_hub_download(
            repo_id=_REPO,
            repo_type='model',
            filename=f'{model_name}/preprocessor.json'
    ), 'r') as f:
        return create_pillow_transforms(json.load(f)['stages'])


@ts_lru_cache()
def _get_dinov2_model(model_name: str):
    """
    Load and cache DINOv2 ONNX model.

    :param model_name: Name of DINOv2 model variant
    :type model_name: str
    :return: Loaded ONNX model
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO,
        repo_type='model',
        filename=f'{model_name}/model.onnx'
    ))


def get_dinov2_embedding(image: ImageTyping, model_name: str = _DEFAULT_MODEL, fmt='embedding', **kwargs):
    """
    Generate embeddings from an image using DINOv2 model.

    This function performs the following steps:

        1. Load and preprocess the image
        2. Run inference using DINOv2 model
        3. Return embeddings in requested format

    :param image: Input image (can be path, URL, PIL Image, etc.)
    :type image: ImageTyping
    :param model_name: Name of DINOv2 model variant to use
    :type model_name: str
    :param fmt: Output format ('embedding', 'pooler_output', or 'last_hidden_state')
    :type fmt: str
    :param kwargs: Additional preprocessing parameters

    :return: Image embeddings in requested format
    :rtype: numpy.ndarray

    :example:
        >>> from realutils.metrics import get_dinov2_embedding
        >>>
        >>> embedding = get_dinov2_embedding('unsplash_0aLd44ICcpg.jpg')
        >>> embedding.shape
        (768,)
    """
    image = load_image(image, force_background='white', mode='RGB')
    preprocessor = _get_preprocessor(model_name)
    input_ = preprocessor(image).astype(np.float32)[None, ...]
    last_hidden_state, pooler_output = _get_dinov2_model(model_name).run(
        ['last_hidden_state', 'pooler_output'],
        {'input': input_},
    )

    return vreplace(fmt, {
        'embedding': pooler_output[0],
        'pooler_output': pooler_output[0],
        'last_hidden_state': last_hidden_state[0],
    })
