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

import json
from typing import List, Union

import numpy as np
from huggingface_hub import hf_hub_download
from imgutils.data import MultiImagesTyping, load_images
from imgutils.preprocess import create_pillow_transforms
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace
from tokenizers import Tokenizer

_REPO_ID = 'deepghs/clip_onnx'
_DEFAULT_MODEL = 'openai/clip-vit-base-patch32'


@ts_lru_cache()
def _open_image_encoder(model_name: str):
    """
    Open and cache the CLIP image encoder model.

    :param model_name: Name of the CLIP model variant
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
    Open and cache the image preprocessor configuration.

    :param model_name: Name of the CLIP model variant
    :type model_name: str

    :return: Configured image preprocessing transforms
    :rtype: callable
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
    Open and cache the CLIP text encoder model.

    :param model_name: Name of the CLIP model variant
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
    Open and cache the text tokenizer.

    :param model_name: Name of the CLIP model variant
    :type model_name: str

    :return: Loaded tokenizer
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
    Get and cache the logit scale factor for the model.

    :param model_name: Name of the CLIP model variant
    :type model_name: str

    :return: Logit scale value
    :rtype: float
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/meta.json',
    ), 'r') as f:
        return json.load(f)['logit_scale']


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
    preprocessor = _open_image_preprocessor(model_name)
    model = _open_image_encoder(model_name)

    images = load_images(images, mode='RGB', force_background='white')
    input_ = np.stack([preprocessor(image) for image in images])
    encodings, embeddings = model.run(['encodings', 'embeddings'], {'pixel_values': input_})
    return vreplace(fmt, {
        'encodings': encodings,
        'embeddings': embeddings,
    })


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
    tokenizer = _open_text_tokenizer(model_name)
    model = _open_text_encoder(model_name)

    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer.encode_batch(texts)
    input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encoded])
    attention_mask = np.stack([np.array(item.attention_mask, dtype=np.int64) for item in encoded])
    encodings, embeddings = model.run(['encodings', 'embeddings'], {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    })
    return vreplace(fmt, {
        'encodings': encodings,
        'embeddings': embeddings,
    })


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
    if not isinstance(images, np.ndarray):
        images = get_clip_image_embedding(images, model_name=model_name, fmt='embeddings')
    images = images / np.linalg.norm(images, axis=-1, keepdims=True)

    if not isinstance(texts, np.ndarray):
        texts = get_clip_text_embedding(texts, model_name=model_name, fmt='embeddings')
    texts = texts / np.linalg.norm(texts, axis=-1, keepdims=True)

    similarities = images @ texts.T
    logits = similarities * np.exp(_get_logit_scale(_DEFAULT_MODEL))
    predictions = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return vreplace(fmt, {
        'similarities': similarities,
        'logits': logits,
        'predictions': predictions,
    })
