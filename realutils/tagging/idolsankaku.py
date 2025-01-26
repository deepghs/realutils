"""
Overview:
    This module provides utilities for image tagging using IdolSankaku taggers.
    It includes functions for loading models, processing images, and extracting tags.

    The module is inspired by the `SmilingWolf/wd-tagger <https://huggingface.co/spaces/SmilingWolf/wd-tagger>`_
    project on Hugging Face.

    .. collapse:: Overview of IdolSankaku (NSFW Warning!!!)

        .. image:: idolsankaku_demo.plot.py.svg
            :align: center

    This is an overall benchmark of all the idolsankaku models:

    .. image:: idolsankaku_benchmark.plot.py.svg
        :align: center
"""
from typing import List, Tuple

import numpy as np
import onnxruntime
import pandas as pd
from PIL import Image
from hbutils.testing.requires.version import VersionInfo
from huggingface_hub import hf_hub_download
from imgutils.data import load_image, ImageTyping
from imgutils.tagging.format import remove_underline
from imgutils.tagging.overlap import drop_overlap_tags
from imgutils.utils import open_onnx_model, vreplace, ts_lru_cache

EXP_REPO = 'deepghs/idolsankaku_tagger_with_embeddings'
EVA02_LARGE_MODEL_DSV3_REPO = "deepghs/idolsankaku-eva02-large-tagger-v1"
SWIN_MODEL_REPO = "deepghs/idolsankaku-swinv2-tagger-v1"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

_IS_SUPPORT = VersionInfo(onnxruntime.__version__) >= '1.17'

MODEL_NAMES = {
    "EVA02_Large": EVA02_LARGE_MODEL_DSV3_REPO,
    "SwinV2": SWIN_MODEL_REPO,
}
_DEFAULT_MODEL_NAME = 'SwinV2'


def _version_support_check(model_name):
    """
    Check if the current onnxruntime version supports the given model.

    :param model_name: The name of the model to check.
    :type model_name: str
    :raises EnvironmentError: If the model is not supported by the current onnxruntime version.
    """
    _ = model_name
    if not _IS_SUPPORT:
        raise EnvironmentError(f'Idolsankaku taggers not supported on onnxruntime {onnxruntime.__version__}, '
                               f'please upgrade it to 1.17+ version.\n'
                               f'If you are running on CPU, use "pip install -U onnxruntime" .\n'
                               f'If you are running on GPU, use "pip install -U onnxruntime-gpu" .')  # pragma: no cover


@ts_lru_cache()
def _get_idolsankaku_model(model_name):
    """
    Load an ONNX model from the Hugging Face Hub.

    :param model_name: The name of the model to load.
    :type model_name: str
    :return: The loaded ONNX model.
    :rtype: ONNXModel
    """
    _version_support_check(model_name)
    return open_onnx_model(hf_hub_download(
        repo_id=EXP_REPO,
        filename=f'{MODEL_NAMES[model_name]}/model.onnx',
    ))


@ts_lru_cache()
def _get_idolsankaku_labels(model_name, no_underline: bool = False) -> Tuple[
    List[str], List[int], List[int], List[int]]:
    """
    Get labels for the IdolSankaku model.

    :param model_name: The name of the model.
    :type model_name: str
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :return: A tuple containing the list of tag names, and lists of indexes for rating, general, and character categories.
    :rtype: Tuple[List[str], List[int], List[int], List[int]]
    """
    df = pd.read_csv(hf_hub_download(
        repo_id=EXP_REPO,
        filename=f'{MODEL_NAMES[model_name]}/selected_tags.csv',
    ))
    name_series = df["name"]
    if no_underline:
        name_series = name_series.map(remove_underline)
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def _mcut_threshold(probs) -> float:
    """
    Compute the Maximum Cut Thresholding (MCut) for multi-label classification.

    This method is based on the paper:
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
    for Multi-label Classification. In 11th International Symposium, IDA 2012
    (pp. 172-183).

    :param probs: Array of probabilities.
    :type probs: numpy.ndarray
    :return: The computed threshold.
    :rtype: float
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


def _prepare_image_for_tagging(image: ImageTyping, target_size: int):
    """
    Prepare an image for tagging by resizing and padding it.

    :param image: The input image.
    :type image: ImageTyping
    :param target_size: The target size for the image.
    :type target_size: int
    :return: The prepared image as a numpy array.
    :rtype: numpy.ndarray
    """
    image = load_image(image, force_background=None, mode=None)
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    try:
        padded_image.paste(image, (pad_left, pad_top), mask=image)
    except ValueError:
        padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1].transpose((2, 0, 1))
    image_array = image_array / 127.5 - 1.0
    return np.expand_dims(image_array, axis=0)


def _postprocess_embedding(
        pred, embedding, logit,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.35,
        general_mcut_enabled: bool = False,
        character_threshold: float = 0.85,
        character_mcut_enabled: bool = False,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt=('rating', 'general', 'character'),
):
    """
    Post-process the embedding and prediction results.

    :param pred: The prediction array.
    :type pred: numpy.ndarray
    :param embedding: The embedding array.
    :type embedding: numpy.ndarray
    :param logit: The logit array.
    :type logit: numpy.ndarray
    :param model_name: The name of the model used.
    :type model_name: str
    :param general_threshold: Threshold for general tags.
    :type general_threshold: float
    :param general_mcut_enabled: Whether to use MCut for general tags.
    :type general_mcut_enabled: bool
    :param character_threshold: Threshold for character tags.
    :type character_threshold: float
    :param character_mcut_enabled: Whether to use MCut for character tags.
    :type character_mcut_enabled: bool
    :param no_underline: Whether to remove underscores from tag names.
    :type no_underline: bool
    :param drop_overlap: Whether to drop overlapping tags.
    :type drop_overlap: bool
    :param fmt: The format of the output.
    :return: The post-processed results.
    """
    assert len(pred.shape) == len(embedding.shape) == 1, \
        f'Both pred and embeddings shapes should be 1-dim, ' \
        f'but pred: {pred.shape!r}, embedding: {embedding.shape!r} actually found.'
    tag_names, rating_indexes, general_indexes, character_indexes = _get_idolsankaku_labels(model_name, no_underline)
    labels = list(zip(tag_names, pred.astype(float)))

    rating = {labels[i][0]: labels[i][1].item() for i in rating_indexes}

    general_names = [labels[i] for i in general_indexes]
    if general_mcut_enabled:
        general_probs = np.array([x[1] for x in general_names])
        general_threshold = _mcut_threshold(general_probs)

    general_res = {x: v.item() for x, v in general_names if v > general_threshold}
    if drop_overlap:
        general_res = drop_overlap_tags(general_res)

    character_names = [labels[i] for i in character_indexes]
    if character_mcut_enabled:
        character_probs = np.array([x[1] for x in character_names])
        character_threshold = _mcut_threshold(character_probs)
        character_threshold = max(0.15, character_threshold)

    character_res = {x: v.item() for x, v in character_names if v > character_threshold}

    return vreplace(
        fmt,
        {
            'rating': rating,
            'general': general_res,
            'character': character_res,
            'tag': {**general_res, **character_res},
            'embedding': embedding.astype(np.float32),
            'prediction': pred.astype(np.float32),
            'logit': logit.astype(np.float32),
        }
    )


def get_idolsankaku_tags(
        image: ImageTyping,
        model_name: str = _DEFAULT_MODEL_NAME,
        general_threshold: float = 0.35,
        general_mcut_enabled: bool = False,
        character_threshold: float = 0.85,
        character_mcut_enabled: bool = False,
        no_underline: bool = False,
        drop_overlap: bool = False,
        fmt=('rating', 'general', 'character'),
):
    """
    Get tags for an image using IdolSankaku taggers.

    This function is similar to the
    `SmilingWolf/wd-tagger <https://huggingface.co/spaces/SmilingWolf/wd-tagger>`_ project on Hugging Face.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The name of the model to use.
    :type model_name: str
    :param general_threshold: The threshold for general tags.
    :type general_threshold: float
    :param general_mcut_enabled: If True, applies MCut thresholding to general tags.
    :type general_mcut_enabled: bool
    :param character_threshold: The threshold for character tags.
    :type character_threshold: float
    :param character_mcut_enabled: If True, applies MCut thresholding to character tags.
    :type character_mcut_enabled: bool
    :param no_underline: If True, replaces underscores in tag names with spaces.
    :type no_underline: bool
    :param drop_overlap: If True, drops overlapping tags.
    :type drop_overlap: bool
    :param fmt: Return format, default is ``('rating', 'general', 'character')``.
        ``embedding`` is also supported for feature extraction.
    :return: Prediction result based on the provided fmt.

    .. note::
        The fmt argument can include the following keys:

        - ``rating``: a dict containing ratings and their confidences
        - ``general``: a dict containing general tags and their confidences
        - ``character``: a dict containing character tags and their confidences
        - ``tag``: a dict containing all tags (including general and character, not including rating) and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logit``: a 1-dim logit of image, before softmax.
        - ``prediction``: a 1-dim prediction result of image

        You can extract embedding of the given image with the follwing code

        >>> from realutils.tagging import get_idolsankaku_tags
        >>>
        >>> embedding = get_idolsankaku_tags('idolsankaku/1.jpg', fmt='embedding')
        >>> embedding.shape
        (1024, )

        This embedding is valuable for constructing indices that enable rapid querying of images based on
        visual features within large-scale datasets.

    Example:
        Here are some images for example

        .. image:: idolsankaku_tiny_demo.plot.py.svg
           :align: center

        >>> from realutils.tagging import get_idolsankaku_tags
        >>>
        >>> rating, general, character = get_idolsankaku_tags('idolsankaku/1.jpg')
        >>> rating
        {'safe': 0.748395562171936, 'questionable': 0.22442740201950073, 'explicit': 0.022273868322372437}
        >>> general
        {'1girl': 0.7476911544799805, 'asian': 0.3681548237800598, 'skirt': 0.8094233274459839, 'solo': 0.44033104181289673, 'blouse': 0.7909733057022095, 'pantyhose': 0.8893758654594421, 'long_hair': 0.7415428161621094, 'brown_hair': 0.4968719780445099, 'sitting': 0.49351146817207336, 'high_heels': 0.41397374868392944, 'outdoors': 0.5279690623283386, 'non_nude': 0.4075928330421448}
        >>> character
        {}
        >>>
        >>> rating, general, character = get_idolsankaku_tags('idolsankaku/7.jpg')
        >>> rating
        {'safe': 0.9750080704689026, 'questionable': 0.0257779061794281, 'explicit': 0.0018109679222106934}
        >>> general
        {'1girl': 0.5759814381599426, 'asian': 0.46296364068984985, 'skirt': 0.9698911905288696, 'solo': 0.6263223886489868, 'female': 0.5258357524871826, 'blouse': 0.8670071959495544, 'twintails': 0.9444552659988403, 'pleated_skirt': 0.8233045935630798, 'miniskirt': 0.8354354500770569, 'long_hair': 0.8752110004425049, 'looking_at_viewer': 0.4927205741405487, 'detached_sleeves': 0.9382797479629517, 'shirt': 0.8463951945304871, 'tie': 0.8901710510253906, 'aqua_hair': 0.9376567006111145, 'armpit': 0.5968506336212158, 'arms_up': 0.9492673873901367, 'sleeveless_blouse': 0.9789504408836365, 'black_thighhighs': 0.41496211290359497, 'sleeveless': 0.9865490198135376, 'default_costume': 0.36392033100128174, 'sleeveless_shirt': 0.9865082502365112, 'very_long_hair': 0.3988983631134033}
        >>> character
        {'hatsune_miku': 0.9460012912750244}
    """

    model = _get_idolsankaku_model(model_name)
    _, _, target_size, _ = model.get_inputs()[0].shape
    input_ = _prepare_image_for_tagging(image, target_size)
    preds, logits, embeddings = model.run(['output', 'logits', 'embedding'], {'input': input_})

    return _postprocess_embedding(
        pred=preds[0],
        embedding=embeddings[0],
        logit=logits[0],
        model_name=model_name,
        general_threshold=general_threshold,
        general_mcut_enabled=general_mcut_enabled,
        character_threshold=character_threshold,
        character_mcut_enabled=character_mcut_enabled,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
        fmt=fmt,
    )
