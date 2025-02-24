from typing import Union, Tuple

import numpy as np
from huggingface_hub import hf_hub_download
from imgutils.data import ImageTyping, load_image
from imgutils.utils import ts_lru_cache, open_onnx_model

from .base import _REPO_ID, _DEFAULT_MODEL, transform, Face


@ts_lru_cache()
def _open_attribute_model(model_name: str):
    session = open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/genderage.onnx'
    ))
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    input_shape = tuple(input_cfg.shape[2:4][::-1])
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, input_shape, output_names


def isf_genderage(image: ImageTyping, face: Union[Face, Tuple[float, float, float, float]],
                  model_name: str = _DEFAULT_MODEL, no_write: bool = False):
    if isinstance(face, Face):
        bbox = face.bbox
    else:
        bbox = face
    image = load_image(image, force_background='white', mode='RGB')
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)

    session, input_name, input_shape, output_names = _open_attribute_model(model_name=model_name)
    assert input_shape[0] == input_shape[1], f'Input shape is not a square - {input_shape!r}.'
    input_size = input_shape[0]
    scale = input_size / (max(w, h) * 1.5)
    aimg, _ = transform(np.array(image), center, input_size, scale, 0)

    blob = aimg.astype(np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW格式

    pred = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    gender = ['F', 'M'][np.argmax(pred[:2]).item()]
    age = int(np.round(pred[2] * 100))
    if isinstance(face, Face) and not no_write:
        face.age = age
        face.gender = gender

    return gender, age


if __name__ == '__main__':
    print(_open_attribute_model(_DEFAULT_MODEL))
