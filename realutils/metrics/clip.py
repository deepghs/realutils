"""
Overview:
    CLIP (Contrastive Language-Image Pre-training) model utilities' module.

    This module provides functions for working with CLIP models, including image and text embedding
    generation and classification. It supports loading ONNX-converted CLIP models from Hugging Face Hub
    and performing inference for both image and text inputs.

    All models and preprocessors are hosted on Huggingface
    repository `deepghs/clip_onnx <https://huggingface.co/deepghs/clip_onnx>`_

    .. image:: clip_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the CLIP models:

    .. image:: clip_image_benchmark.plot.py.svg
        :align: center

    .. image:: clip_text_benchmark.plot.py.svg
        :align: center

"""

from typing import List, Union

import numpy as np
from imgutils.data import MultiImagesTyping
from imgutils.generic import clip_image_encode, clip_text_encode, clip_predict

_REPO_ID = 'deepghs/clip_onnx'
_DEFAULT_MODEL = 'openai/clip-vit-base-patch32'


def get_clip_image_embedding(images: MultiImagesTyping, model_name: str = _DEFAULT_MODEL, fmt='embeddings'):
    """
    Generate CLIP embeddings for input images.

    :param images: Input images to encode
    :type images: MultiImagesTyping
    :param model_name: Name of the CLIP model to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or 'encodings')

    :return: Image embeddings or encodings based on fmt parameter

    :example:
        >>> from realutils.metrics.clip import get_clip_image_embedding
        >>>
        >>> # one image
        >>> emb = get_clip_image_embedding('xlip/1.jpg')
        >>> emb.shape, emb.dtype
        ((1, 512), dtype('float32'))
        >>>
        >>> # multiple images
        >>> emb = get_clip_image_embedding(['xlip/1.jpg', 'xlip/2.jpg'])
        >>> emb.shape, emb.dtype
        ((2, 512), dtype('float32'))
    """
    return clip_image_encode(
        images=images,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )


def get_clip_text_embedding(texts: Union[str, List[str]], model_name: str = _DEFAULT_MODEL, fmt='embeddings'):
    """
    Generate CLIP embeddings for input texts.

    :param texts: Input text or list of texts to encode
    :type texts: Union[str, List[str]]
    :param model_name: Name of the CLIP model to use
    :type model_name: str
    :param fmt: Output format ('embeddings' or 'encodings')

    :return: Text embeddings or encodings based on fmt parameter

    :example:
        >>> from realutils.metrics.clip import get_clip_text_embedding
        >>>
        >>> # one text
        >>> emb = get_clip_text_embedding('a photo of a cat')
        >>> emb.shape, emb.dtype
        ((1, 512), dtype('float32'))
        >>>
        >>> # multiple texts
        >>> emb = get_clip_text_embedding([
        ...     'a photo of a cat',
        ...     'a photo of a dog',
        ...     'a photo of a human',
        ... ])
        >>> emb.shape, emb.dtype
        ((3, 512), dtype('float32'))
    """
    return clip_text_encode(
        texts=texts,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )


def classify_with_clip(
        images: Union[MultiImagesTyping, np.ndarray],
        texts: Union[List[str], str, np.ndarray],
        model_name: str = _DEFAULT_MODEL,
        fmt='predictions',
):
    """
    Perform classification using CLIP model by comparing image and text embeddings.

    :param images: Input images or pre-computed image embeddings
    :type images: Union[MultiImagesTyping, numpy.ndarray]
    :param texts: Input texts or pre-computed text embeddings
    :type texts: Union[List[str], str, numpy.ndarray]
    :param model_name: Name of the CLIP model to use
    :type model_name: str
    :param fmt: Output format ('predictions', 'similarities', or 'logits')

    :return: Classification results based on fmt parameter

    :example:
        >>> from realutils.metrics.clip import classify_with_clip
        >>>
        >>> classify_with_clip(
        ...     images=[
        ...         'xlip/1.jpg',
        ...         'xlip/2.jpg'
        ...     ],
        ...     texts=[
        ...         'a photo of a cat',
        ...         'a photo of a dog',
        ...         'a photo of a human',
        ...     ],
        ... )
        array([[0.98039913, 0.00506729, 0.01453355],
               [0.05586662, 0.02006196, 0.92407143]], dtype=float32)
    """
    return clip_predict(
        images=images,
        texts=texts,
        repo_id=_REPO_ID,
        model_name=model_name,
        fmt=fmt,
    )
