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
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/image_encode.onnx',
    ))


@ts_lru_cache()
def _open_image_preprocessor(model_name: str):
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/preprocessor.json',
    ), 'r') as f:
        return create_pillow_transforms(json.load(f)['stages'])


@ts_lru_cache()
def _open_text_encoder(model_name: str):
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/text_encode.onnx',
    ))


@ts_lru_cache()
def _open_text_tokenizer(model_name: str):
    return Tokenizer.from_file(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/tokenizer.json',
    ))


@ts_lru_cache()
def _get_logit_scale(model_name: str):
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/meta.json',
    ), 'r') as f:
        meta_info = json.load(f)
        return meta_info['logit_scale'], meta_info['logit_bias']


def get_siglip_image_embedding(images: MultiImagesTyping, model_name: str = _DEFAULT_MODEL, fmt='embeddings'):
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


if __name__ == '__main__':
    print(classify_with_siglip(
        images=[
            'test_image.jpg',
            'test/testfile/idolsankaku/3.jpg',
        ],
        texts=[
            'a photo of a cat',
            'a photo of a dog',
            # "Hello, this is a test sentence.",
            'a photo of a human',
        ],
    ))
