"""
This module provides functionality for generating embeddings from images using the DINOv2 model.
It includes utilities for image preprocessing and model inference using ONNX runtime.

The module supports different DINOv2 model variants and provides configurable preprocessing options.
"""

import copy
import json

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from imgutils.data import ImageTyping, load_image
from imgutils.utils import open_onnx_model, ts_lru_cache, vreplace

_DEFAULT = object()
_DEFAULT_SIZE = {"shortest_edge": 256}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def _dinov2_preprocess_image(
        image: Image.Image,
        do_resize: bool = True,
        size: dict = _DEFAULT,
        resample: int = 3,  # BICUBIC
        do_center_crop: bool = True,
        crop_size: dict = _DEFAULT,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: list = _DEFAULT,
        image_std: list = _DEFAULT,
        do_convert_rgb: bool = True
) -> np.ndarray:
    """
    Preprocess an image according to DINOv2 model requirements.

    This function performs several preprocessing steps:
    1. RGB conversion (optional)
    2. Resizing (optional)
    3. Center cropping (optional)
    4. Pixel value rescaling (optional)
    5. Channel-first conversion
    6. Normalization (optional)

    :param image: Input PIL Image
    :type image: PIL.Image.Image
    :param do_resize: Whether to resize the image
    :type do_resize: bool
    :param size: Resize configuration dict with either 'shortest_edge' or ('height', 'width')
    :type size: dict
    :param resample: PIL resample method, default is BICUBIC
    :type resample: int
    :param do_center_crop: Whether to perform center cropping
    :type do_center_crop: bool
    :param crop_size: Crop size configuration dict with 'height' and 'width'
    :type crop_size: dict
    :param do_rescale: Whether to rescale pixel values
    :type do_rescale: bool
    :param rescale_factor: Factor to rescale pixel values
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image
    :type do_normalize: bool
    :param image_mean: Mean values for normalization
    :type image_mean: list
    :param image_std: Standard deviation values for normalization
    :type image_std: list
    :param do_convert_rgb: Whether to convert image to RGB
    :type do_convert_rgb: bool

    :return: Preprocessed image as numpy array
    :rtype: numpy.ndarray
    """
    if do_convert_rgb and image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    if do_resize:
        if size is _DEFAULT:
            size = _DEFAULT_SIZE
        if 'shortest_edge' in size:
            h, w = img_array.shape[:2]
            shortest_edge = min(h, w)
            scale = size['shortest_edge'] / shortest_edge
            new_h = int(h * scale)
            new_w = int(w * scale)
        elif 'width' in size and 'height' in size:
            new_h, new_w = size['height'], size['width']
        else:
            raise ValueError(f'Invalid resize - {size!r}.')  # pragma: no cover
        image = Image.fromarray(img_array)
        image = image.resize((new_w, new_h), resample=resample)
        img_array = np.array(image)

    if do_center_crop:
        if crop_size is _DEFAULT:
            crop_size = _DEFAULT_CROP_SIZE
        h, w = img_array.shape[:2]
        crop_h = crop_size['height']
        crop_w = crop_size['width']
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        img_array = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]

    if do_rescale:
        img_array = img_array * rescale_factor

    img_array = img_array.transpose(2, 0, 1)
    if do_normalize:
        if image_mean is _DEFAULT:
            image_mean = _DEFAULT_MEAN
        if image_std is _DEFAULT:
            image_std = _DEFAULT_STD
        mean = np.array(image_mean).reshape(-1, 1, 1)
        std = np.array(image_std).reshape(-1, 1, 1)
        img_array = (img_array - mean) / std

    return img_array


_REPO = 'deepghs/dinov2_onnx'
_DEFAULT_MODEL = 'facebook/dinov2-base'


@ts_lru_cache()
def _get_preprocess_config(model_name: str):
    """
    Get preprocessing configuration for specified DINOv2 model variant.

    :param model_name: Name of DINOv2 model variant
    :type model_name: str
    :return: Preprocessing configuration dictionary
    :rtype: dict
    """
    with open(hf_hub_download(
            repo_id=_REPO,
            repo_type='model',
            filename=f'{model_name}/preprocess.json'
    ), 'r') as f:
        return json.load(f)


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
    """
    image = load_image(image, force_background='white', mode='RGB')
    preprocess_config = copy.deepcopy(_get_preprocess_config(model_name))
    assert preprocess_config["image_processor_type"] == "BitImageProcessor", \
        f'Unsupported preprocessor - {preprocess_config["image_processor_type"]!r}'
    del preprocess_config["image_processor_type"]

    data = _dinov2_preprocess_image(image, **{**preprocess_config, **kwargs})
    data = data.astype(np.float32)
    last_hidden_state, pooler_output = _get_dinov2_model(model_name).run(
        ['last_hidden_state', 'pooler_output'],
        {'input': data[None, ...]},
    )

    return vreplace(fmt, {
        'embedding': pooler_output[0],
        'pooler_output': pooler_output[0],
        'last_hidden_state': last_hidden_state[0],
    })
