import os.path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime
import torch.nn
from PIL import Image
from ditk import logging
from transformers import CLIPModel, CLIPProcessor

from zoo.testings import get_testfile
from zoo.utils import onnx_optimize


class CLIPImageToEmbedding(torch.nn.Module):
    def __init__(self, clip_model: CLIPModel, attention_mask: torch.Tensor, input_ids: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('attention_mask', attention_mask)
        self.attention_mask: torch.Tensor
        self.register_buffer('input_ids', input_ids)
        self.input_ids: torch.Tensor

    def forward(self, pixel_values: torch.Tensor):
        voutput = self.clip_model(
            pixel_values=pixel_values,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
        )
        return voutput.vision_model_output.pooler_output, voutput.image_embeds


class CLIPTextToEmbedding(torch.nn.Module):
    def __init__(self, clip_model: CLIPModel, pixel_values: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('pixel_values', pixel_values)
        self.pixel_values: torch.Tensor

    def forward(self, attention_mask, input_ids):
        toutput = self.clip_model(
            pixel_values=self.pixel_values,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        return toutput.text_model_output.pooler_output, toutput.text_embeds


def get_clip_model(model_name: str):
    logging.info(f'Loading model {model_name!r} ...')
    model = CLIPModel.from_pretrained(model_name)
    return model


def get_dummy_input(model_name: str):
    logging.info(f'Getting dummy input of model {model_name!r} ...')
    processor = CLIPProcessor.from_pretrained(model_name)

    image = Image.open(get_testfile('clip_cat.jpg'))
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    return inputs


def export_image_to_onnx(model_raw, dummy_input, export_onnx_file: str = 'test_clip_image.onnx',
                         no_optimize: bool = False):
    logging.info('Wrapping image encoding model ...')
    model_image = CLIPImageToEmbedding(
        model_raw,
        attention_mask=dummy_input.attention_mask,
        input_ids=dummy_input.input_ids,
    )
    with torch.no_grad():
        expected_encodings, expected_embeddings = model_image(dummy_input.pixel_values)
    logging.info(f'Expected encodings, shape: {expected_encodings.shape}, dtype: {expected_encodings.shape}')
    logging.info(f'Expected embeddings, shape: {expected_embeddings.shape}, dtype: {expected_embeddings.shape}')

    with TemporaryDirectory() as td:
        temp_onnx_file = os.path.join(td, 'onnx_image_temp.onnx')
        torch.onnx.export(
            model_image,
            dummy_input.pixel_values,
            temp_onnx_file,
            input_names=['pixel_values'],
            output_names=['encodings', 'embeddings'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'encodings': {0: 'batch_size'},
                'embeddings': {0: 'batch_size'},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            custom_opsets=None,
        )

        logging.info(f'Optimizing onnx file {temp_onnx_file!r} ...')
        model = onnx.load(temp_onnx_file)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        logging.info(f'Exporting image model to onnx file {export_onnx_file!r} ...')
        output_model_dir, _ = os.path.split(export_onnx_file)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, export_onnx_file)

    logging.info(f'Validating exported onnx file {export_onnx_file!r} ...')
    session = onnxruntime.InferenceSession(export_onnx_file)
    encodings, embedding = session.run(
        ['encodings', 'embeddings'],
        {'pixel_values': dummy_input.pixel_values.numpy()},
    )
    logging.info(f'ONNX encodings, shape: {encodings.shape}, dtype: {encodings.dtype}')
    logging.info(f'ONNX embeddings, shape: {embedding.shape}, dtype: {embedding.dtype}')
    logging.info('Comparing outputs ...')
    np.testing.assert_allclose(expected_encodings.numpy(), encodings, atol=1e-4, rtol=1e-2)
    np.testing.assert_allclose(expected_embeddings.numpy(), embedding, atol=1e-5, rtol=1e-2)


def export_text_to_onnx(model_raw, dummy_input, export_onnx_file: str = 'test_clip_text.onnx',
                        no_optimize: bool = False):
    logging.info('Wrapping text encoding model ...')
    model_text = CLIPTextToEmbedding(model_raw, pixel_values=dummy_input.pixel_values)

    with torch.no_grad():
        expected_encodings, expected_embeddings = model_text(dummy_input.attention_mask, dummy_input.input_ids)
    logging.info(f'Expected encodings, shape: {expected_encodings.shape}, dtype: {expected_encodings.shape}')
    logging.info(f'Expected embeddings, shape: {expected_embeddings.shape}, dtype: {expected_embeddings.shape}')

    with TemporaryDirectory() as td:
        temp_onnx_file = os.path.join(td, 'onnx_text_temp.onnx')
        torch.onnx.export(
            model_text,
            (dummy_input.attention_mask, dummy_input.input_ids),
            temp_onnx_file,
            input_names=['attention_mask', 'input_ids'],
            output_names=['encodings', 'embeddings'],
            dynamic_axes={
                'attention_mask': {0: 'batch_size', 1: 'text_length'},
                'input_ids': {0: 'batch_size', 1: 'text_length'},
                'encodings': {0: 'batch_size'},
                'embeddings': {0: 'batch_size'},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            custom_opsets=None,
        )

        logging.info(f'Optimizing onnx file {temp_onnx_file!r} ...')
        model = onnx.load(temp_onnx_file)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        logging.info(f'Exporting text model to onnx file {export_onnx_file!r} ...')
        output_model_dir, _ = os.path.split(export_onnx_file)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, export_onnx_file)

    logging.info(f'Validating exported onnx file {export_onnx_file!r} ...')
    session = onnxruntime.InferenceSession(export_onnx_file)
    encodings, embedding = session.run(
        ['encodings', 'embeddings'],
        {
            'attention_mask': dummy_input.attention_mask.numpy(),
            'input_ids': dummy_input.input_ids.numpy(),
        },
    )
    logging.info(f'ONNX encodings, shape: {encodings.shape}, dtype: {encodings.dtype}')
    logging.info(f'ONNX embeddings, shape: {embedding.shape}, dtype: {embedding.dtype}')
    logging.info('Comparing outputs ...')
    np.testing.assert_allclose(expected_encodings.numpy(), encodings, atol=1e-4, rtol=1e-2)
    np.testing.assert_allclose(expected_embeddings.numpy(), embedding, atol=1e-5, rtol=1e-2)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    model_name = 'openai/clip-vit-base-patch32'
    dummy_input = get_dummy_input(model_name)
    model_raw = get_clip_model(model_name)
    # with torch.no_grad():
    #     print(model_raw.logit_scale)
    #     print(model_raw.logit_scale.exp())

    # export_image_to_onnx(
    #     model_raw=model_raw,
    #     dummy_input=dummy_input,
    # )
    export_text_to_onnx(
        model_raw=model_raw,
        dummy_input=dummy_input,
    )
