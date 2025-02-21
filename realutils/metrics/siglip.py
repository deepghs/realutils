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

from typing import List, Union, Any

import numpy as np
from imgutils.data import MultiImagesTyping
from imgutils.generic import siglip_image_encode, siglip_text_encode, siglip_predict

_REPO_ID = 'deepghs/siglip_onnx'
_DEFAULT_MODEL = 'google/siglip-base-patch16-256-multilingual'


def get_siglip_image_embedding(images: MultiImagesTyping, model_name: str = _DEFAULT_MODEL, fmt: Any = 'embeddings'):
    """
    Generate embeddings for input images using the SigLIP model.

    :param images: Input images in various supported formats
    :type images: MultiImagesTyping
    :param model_name: Name of the SigLIP model variant to use
    :type model_name: str
    :param fmt: Output format, either 'encodings' or 'embeddings'
    :type fmt: Any

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
    return siglip_image_encode(
        images=images,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )


def get_siglip_text_embedding(texts: Union[str, List[str]], model_name: str = _DEFAULT_MODEL, fmt: Any = 'embeddings'):
    """
    Generate embeddings for input texts using the SigLIP model.

    :param texts: Input text or list of texts
    :type texts: Union[str, List[str]]
    :param model_name: Name of the SigLIP model variant to use
    :type model_name: str
    :param fmt: Output format, either 'encodings' or 'embeddings'
    :type fmt: Any

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
    return siglip_text_encode(
        texts=texts,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )


def classify_with_siglip(
        images: Union[MultiImagesTyping, np.ndarray],
        texts: Union[List[str], str, np.ndarray],
        model_name: str = _DEFAULT_MODEL,
        fmt: Any = 'predictions',
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
    :type fmt: Any

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
    return siglip_predict(
        images=images,
        texts=texts,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )
