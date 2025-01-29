import json
import os.path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime
import torch.nn
from PIL import Image
from ditk import logging
from huggingface_hub import hf_hub_download
from imgutils.preprocess import create_transforms_from_transformers, parse_pillow_transforms
from thop import clever_format, profile
from tokenizers import Tokenizer
from transformers import SiglipImageProcessor, SiglipModel, SiglipProcessor

from zoo.testings import get_testfile
from zoo.utils import onnx_optimize


class SiglipImageToEmbedding(torch.nn.Module):
    def __init__(self, siglip_model: SiglipModel, input_ids: torch.Tensor):
        super().__init__()
        self.siglip_model = siglip_model
        self.register_buffer('input_ids', input_ids)
        self.input_ids: torch.Tensor

    def forward(self, pixel_values: torch.Tensor):
        voutput = self.siglip_model(
            pixel_values=pixel_values,
            input_ids=self.input_ids,
        )
        return voutput.vision_model_output.pooler_output, voutput.image_embeds


class SiglipTextToEmbedding(torch.nn.Module):
    def __init__(self, siglip_model: SiglipModel, pixel_values: torch.Tensor):
        super().__init__()
        self.siglip_model = siglip_model
        self.register_buffer('pixel_values', pixel_values)
        self.pixel_values: torch.Tensor

    def forward(self, input_ids):
        toutput = self.siglip_model(
            pixel_values=self.pixel_values,
            input_ids=input_ids,
        )
        return toutput.text_model_output.pooler_output, toutput.text_embeds


def get_siglip_model(model_name: str):
    logging.info(f'Loading model {model_name!r} ...')
    model = SiglipModel.from_pretrained(model_name)
    return model


def get_dummy_input(model_name: str):
    logging.info(f'Getting dummy input of model {model_name!r} ...')
    processor = SiglipProcessor.from_pretrained(model_name)

    image = Image.open(get_testfile('clip_cat.jpg'))
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    return processor, inputs


def export_image_to_onnx(model_raw, dummy_input, export_onnx_file: str = 'test_siglip_image.onnx',
                         no_optimize: bool = False):
    logging.info('Wrapping image encoding model ...')
    model_image = SiglipImageToEmbedding(
        model_raw,
        input_ids=dummy_input.input_ids,
    )
    with torch.no_grad():
        expected_encodings, expected_embeddings = model_image(dummy_input.pixel_values)
    logging.info(f'Expected encodings, shape: {expected_encodings.shape}, dtype: {expected_encodings.shape}')
    logging.info(f'Expected embeddings, shape: {expected_embeddings.shape}, dtype: {expected_embeddings.shape}')

    logging.info('Profiling model ...')
    with torch.no_grad():
        flops, params = profile(model_raw.vision_model, inputs=(dummy_input.pixel_values,))
    s_flops, s_params = clever_format([flops, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_flops}')

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

    return (flops, params), dummy_input.pixel_values.shape[2], (encodings.shape[-1], embedding.shape[-1])


def export_text_to_onnx(model_raw, dummy_input, export_onnx_file: str = 'test_siglip_text.onnx',
                        no_optimize: bool = False):
    logging.info('Wrapping text encoding model ...')
    model_text = SiglipTextToEmbedding(model_raw, pixel_values=dummy_input.pixel_values)

    with torch.no_grad():
        expected_encodings, expected_embeddings = model_text(dummy_input.input_ids)
    logging.info(f'Expected encodings, shape: {expected_encodings.shape}, dtype: {expected_encodings.shape}')
    logging.info(f'Expected embeddings, shape: {expected_embeddings.shape}, dtype: {expected_embeddings.shape}')

    logging.info('Profiling model ...')
    with torch.no_grad():
        flops, params = profile(model_raw.text_model, inputs=(dummy_input.input_ids,))
    s_flops, s_params = clever_format([flops, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_flops}')

    with TemporaryDirectory() as td:
        temp_onnx_file = os.path.join(td, 'onnx_text_temp.onnx')
        torch.onnx.export(
            model_text,
            (dummy_input.input_ids,),
            temp_onnx_file,
            input_names=['input_ids'],
            output_names=['encodings', 'embeddings'],
            dynamic_axes={
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
            'input_ids': dummy_input.input_ids.numpy(),
        },
    )
    logging.info(f'ONNX encodings, shape: {encodings.shape}, dtype: {encodings.dtype}')
    logging.info(f'ONNX embeddings, shape: {embedding.shape}, dtype: {embedding.dtype}')
    logging.info('Comparing outputs ...')
    np.testing.assert_allclose(expected_encodings.numpy(), encodings, atol=1e-4, rtol=1e-2)
    np.testing.assert_allclose(expected_embeddings.numpy(), embedding, atol=1e-5, rtol=1e-2)

    return (flops, params), (encodings.shape[-1], embedding.shape[-1])


def export_image_preprocessor(preprocessor: SiglipImageProcessor,
                              preprocessor_file: str = 'test_siglip_image_preprocessor.json'):
    if os.path.dirname(preprocessor_file):
        os.makedirs(os.path.dirname(preprocessor_file), exist_ok=True)

    pillow_trans = create_transforms_from_transformers(preprocessor)
    logging.info('Extracted preprocessor stages:\n'
                 f'{pillow_trans}')
    logging.info(f'Writing to {preprocessor_file!r} ...')
    with open(preprocessor_file, 'w') as f:
        json.dump({
            'stages': parse_pillow_transforms(pillow_trans),
        }, f, indent=4, sort_keys=True)


def export_text_tokenizer(repo_id: str,
                          tokenizer_file: str = 'test_siglip_text_tokenizer.json'):
    if os.path.dirname(tokenizer_file):
        os.makedirs(os.path.dirname(tokenizer_file), exist_ok=True)

    tokenizer = Tokenizer.from_file(hf_hub_download(
        repo_id=repo_id,
        repo_type='model',
        filename='tokenizer.json',
    ))
    tokenizer.enable_padding(
        direction='right',
        pad_id=1,
        pad_type_id=0,
        pad_token="</s>",
    )

    with TemporaryDirectory() as td:
        logging.info(f'Saving full pretrained to {td!r} ...')
        tokenizer.save(os.path.join(td, 'tokenizer.json'))

        src_tokenizer_file = os.path.join(td, 'tokenizer.json')
        logging.info(f'Copying {src_tokenizer_file!r} to {tokenizer_file!r} ...')
        shutil.copyfile(src_tokenizer_file, tokenizer_file)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    repo_id = "google/siglip-base-patch16-256-multilingual"

    model_raw = get_siglip_model(repo_id)
    processor, dummy_input = get_dummy_input(repo_id)

    # export_text_to_onnx(model_raw, dummy_input)
    # export_image_to_onnx(model_raw, dummy_input)
    export_image_preprocessor(processor.image_processor)
    export_text_tokenizer(repo_id)