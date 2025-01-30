"""
Overview:
    This module provides functionality for image-text matching using the SigLIP (Sigmoid Loss Pre-training of Image-text Pairs) model.
    It includes functions for encoding images and text into embeddings, and performing classification using these embeddings.
    The module uses ONNX models downloaded from Hugging Face Hub and provides caching mechanisms for improved performance.

    All models and preprocessors are hosted on Huggingface
    repository `deepghs/siglip_onnx <https://huggingface.co/deepghs/siglip_onnx>`_

    .. image:: siglip_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the SigLIP models:

    .. image:: siglip_image_benchmark.plot.py.svg
        :align: center

    .. image:: siglip_text_benchmark.plot.py.svg
        :align: center
"""

import json
from typing import List, Union

import numpy as np
from huggingface_hub import hf_hub_download
from imgutils.data import MultiImagesTyping, load_images
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace, sigmoid
from tokenizers import Tokenizer

_REPO_ID = 'deepghs/siglip_onnx'
_DEFAULT_MODEL = 'google/siglip-base-patch16-256-multilingual'


@ts_lru_cache()
def _open_image_encoder(model_name: str):
    """
    Open and cache the ONNX image encoder model.

    :param model_name: Name of the SigLIP model variant
    :type model_name: str
    :return: Loaded ONNX model for image encoding
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/image_encode.onnx',
    ))


@ts_lru_cache()
def _open_image_preprocessor(model_name: str):
    """
    Load and cache the image preprocessing pipeline configuration.

    :param model_name: Name of the SigLIP model variant
    :type model_name: str
    :return: Configured image preprocessing transforms
    :rtype: Callable
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/preprocessor.json',
    ), 'r') as f:
        return create_pillow_transforms(json.load(f)['stages'])


@ts_lru_cache()
def _open_text_encoder(model_name: str):
    """
    Open and cache the ONNX text encoder model.

    :param model_name: Name of the SigLIP model variant
    :type model_name: str
    :return: Loaded ONNX model for text encoding
    :rtype: ONNXModel
    """
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/text_encode.onnx',
    ))


@ts_lru_cache()
def _open_text_tokenizer(model_name: str):
    """
    Load and cache the text tokenizer.

    :param model_name: Name of the SigLIP model variant
    :type model_name: str
    :return: Initialized tokenizer
    :rtype: Tokenizer
    """
    return Tokenizer.from_file(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/tokenizer.json',
    ))


@ts_lru_cache()
def _get_logit_scale(model_name: str):
    """
    Get the logit scale and bias parameters from model metadata.

    :param model_name: Name of the SigLIP model variant
    :type model_name: str
    :return: Tuple of logit scale and bias values
    :rtype: tuple[float, float]
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/meta.json',
    ), 'r') as f:
        meta_info = json.load(f)
        return meta_info['logit_scale'], meta_info['logit_bias']


def get_siglip_image_embedding(images: MultiImagesTyping, model_name: str = _DEFAULT_MODEL, fmt='embeddings'):
    """
    Generate embeddings for input images using the SigLIP model.

    :param images: Input images in various supported formats
    :type images: MultiImagesTyping
    :param model_name: Name of the SigLIP model variant to use
    :type model_name: str
    :param fmt: Output format, either 'encodings' or 'embeddings'

    :return: Image embeddings or encodings based on fmt parameter

    :example:
        >>> from realutils.metrics.siglip import get_siglip_image_embedding
        >>>
        >>> # one image
        >>> emb = get_siglip_image_embedding('xlip/1.jpg')
        >>> emb.shape, emb.dtype
        ((1, 768), dtype('float32'))
        >>>
        >>> # multiple images
        >>> emb = get_siglip_image_embedding(['xlip/1.jpg', 'xlip/2.jpg'])
        >>> emb.shape, emb.dtype
        ((2, 768), dtype('float32'))
    """
    preprocessor = _open_image_preprocessor(model_name)
    model = _open_image_encoder(model_name)

    images = load_images(images, mode='RGB', force_background='white')
    input_ = np.stack([preprocessor(image) for image in images])
    encodings, embeddings = model.run(['encodings', 'embeddings'], {'pixel_values': input_})
    return vreplace(fmt, {
        'encodings': encodings,
        'embeddings': embeddings,
    })


def get_siglip_text_embedding(texts: Union[str, List[str]], model_name: str = _DEFAULT_MODEL, fmt='embeddings'):
    """
    Generate embeddings for input texts using the SigLIP model.

    :param texts: Input text or list of texts
    :type texts: Union[str, List[str]]
    :param model_name: Name of the SigLIP model variant to use
    :type model_name: str
    :param fmt: Output format, either 'encodings' or 'embeddings'

    :return: Text embeddings or encodings based on fmt parameter

    :example:
        >>> from realutils.metrics.siglip import get_siglip_text_embedding
        >>>
        >>> # one text
        >>> emb = get_siglip_text_embedding('a photo of a cat')
        >>> emb.shape, emb.dtype
        ((1, 768), dtype('float32'))
        >>>
        >>> # multiple texts
        >>> emb = get_siglip_text_embedding([
        ...     'a photo of a cat',
        ...     'a photo of 2 cats',
        ...     'a photo of a dog',
        ...     'a photo of a woman',
        ... ])
        >>> emb.shape, emb.dtype
        ((4, 768), dtype('float32'))
    """
    tokenizer = _open_text_tokenizer(model_name)
    model = _open_text_encoder(model_name)

    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer.encode_batch(texts)
    input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encoded])
    encodings, embeddings = model.run(['encodings', 'embeddings'], {
        'input_ids': input_ids,
    })
    return vreplace(fmt, {
        'encodings': encodings,
        'embeddings': embeddings,
    })


def classify_with_siglip(
        images: Union[MultiImagesTyping, np.ndarray],
        texts: Union[List[str], str, np.ndarray],
        model_name: str = _DEFAULT_MODEL,
        fmt='predictions',
):
    """
    Perform image-text classification using the SigLIP model.

    :param images: Input images or pre-computed image embeddings
    :type images: Union[MultiImagesTyping, numpy.ndarray]
    :param texts: Input texts or pre-computed text embeddings
    :type texts: Union[List[str], str, numpy.ndarray]
    :param model_name: Name of the SigLIP model variant to use
    :type model_name: str
    :param fmt: Output format, one of 'similarities', 'logits', or 'predictions'

    :return: Classification results in specified format

    :example:
        >>> from realutils.metrics.siglip import classify_with_siglip
        >>>
        >>> classify_with_siglip(
        ...     images=[
        ...         'xlip/1.jpg',
        ...         'xlip/2.jpg',
        ...     ],
        ...     texts=[
        ...         'a photo of a cat',
        ...         'a photo of 2 cats',
        ...         'a photo of 2 dogs',
        ...         'a photo of a woman',
        ...     ],
        ... )
        array([[1.3782851e-03, 2.7010253e-01, 9.7517688e-05, 3.6702781e-09],
               [3.3248414e-06, 2.2294161e-07, 1.9753381e-09, 2.2561464e-06]],
              dtype=float32)
    """
    if not isinstance(images, np.ndarray):
        images = get_siglip_image_embedding(images, model_name=model_name, fmt='embeddings')
    images = images / np.linalg.norm(images, axis=-1, keepdims=True)

    if not isinstance(texts, np.ndarray):
        texts = get_siglip_text_embedding(texts, model_name=model_name, fmt='embeddings')
    texts = texts / np.linalg.norm(texts, axis=-1, keepdims=True)

    similarities = images @ texts.T
    logit_scale, logit_bias = _get_logit_scale(_DEFAULT_MODEL)
    logits = similarities * np.exp(logit_scale) + logit_bias
    predictions = sigmoid(logits)

    return vreplace(fmt, {
        'similarities': similarities,
        'logits': logits,
        'predictions': predictions,
    })
