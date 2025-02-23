import copy
import datetime
import json
import os.path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch.nn
from PIL import Image
from ditk import logging
from hbutils.string import plural_word
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_download
from imgutils.preprocess import create_transforms_from_transformers, parse_pillow_transforms
from natsort import natsorted
from thop import clever_format, profile
from tokenizers import Tokenizer
from tqdm import tqdm
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
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image,
                       return_tensors="pt", padding='max_length')
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
        length=64,
        direction='right',
        pad_id=1,
        pad_type_id=0,
        pad_token="</s>",
    )

    with TemporaryDirectory() as td:
        logging.info(f'Saving full pretrained to {td!r} ...')
        src_tokenizer_file = os.path.join(td, 'tokenizer.json')
        tokenizer.save(src_tokenizer_file)

        logging.info('Validating tokenizer ...')
        actual_tokenizer = Tokenizer.from_file(src_tokenizer_file)
        image = Image.open(get_testfile('clip_cat.jpg'))
        texts_to_test = [
            "a photo of a cat",
            "a photo of a dog",
            "Hello, this is a test sentence.",
            'a photo of a human',
        ]
        processor = SiglipProcessor.from_pretrained(repo_id)
        inputs = processor(text=texts_to_test, images=image,
                           return_tensors="pt", padding='max_length')
        expected_input_ids = inputs.input_ids.numpy()
        encoded = actual_tokenizer.encode_batch(texts_to_test)
        actual_input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encoded])
        np.testing.assert_allclose(expected_input_ids, actual_input_ids)

        logging.info(f'Copying {src_tokenizer_file!r} to {tokenizer_file!r} ...')
        shutil.copyfile(src_tokenizer_file, tokenizer_file)


def sync(repository: str = 'deepghs/siglip_onnx'):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_detached_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=False)
        attr_lines = hf_fs.read_text(f'{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(f'{repository}/.gitattributes', os.linesep.join(attr_lines))

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
    ):
        df_models = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
        ))
        d_models = {item['name']: item for item in df_models.to_dict('records')}
    else:
        d_models = {}

    _KNOWN_MODELS = [
        'google/siglip-base-patch16-256-multilingual',
        'google/siglip-base-patch16-224',
        'google/siglip-base-patch16-512',
        'google/siglip-large-patch16-256',
        'google/siglip-base-patch16-384',
        'google/siglip-large-patch16-384',
        'google/siglip-base-patch16-256',

        'google/siglip-so400m-patch14-384',
        'google/siglip-so400m-patch14-224',
        'google/siglip-so400m-patch16-256-i18n',
    ]
    for model_repo_id in tqdm(_KNOWN_MODELS, desc='Exporting Models'):
        if not hf_client.repo_exists(repo_id=model_repo_id, repo_type='model'):
            logging.warn(f'Repo {model_repo_id!r} not exist, skipped.')
            continue

        if hf_client.file_exists(
                repo_id=repository,
                repo_type='model',
                filename=f'{model_repo_id}/image_encode.onnx',
        ):
            logging.warn(f'Model {model_repo_id!r} already exported, skipped.')
            continue

        with TemporaryDirectory() as upload_dir:
            logging.info(f'Exporting model {model_repo_id!r} ...')
            os.makedirs(os.path.join(upload_dir, model_repo_id), exist_ok=True)

            repo_created_at = hf_client.repo_info(repo_id=model_repo_id, repo_type='model').created_at.timestamp()

            processor, dummy_input = get_dummy_input(model_repo_id)
            model_raw = get_siglip_model(model_repo_id)

            (img_flops, img_params), input_img_size, (img_encoding_width, img_embedding_width) = export_image_to_onnx(
                model_raw=model_raw,
                dummy_input=dummy_input,
                export_onnx_file=os.path.join(upload_dir, model_repo_id, 'image_encode.onnx'),
            )
            (text_flops, text_params), (text_encoding_width, text_embedding_width) = export_text_to_onnx(
                model_raw=model_raw,
                dummy_input=dummy_input,
                export_onnx_file=os.path.join(upload_dir, model_repo_id, 'text_encode.onnx'),
            )
            export_image_preprocessor(
                preprocessor=processor.image_processor,
                preprocessor_file=os.path.join(upload_dir, model_repo_id, 'preprocessor.json')
            )
            export_text_tokenizer(
                repo_id=model_repo_id,
                tokenizer_file=os.path.join(upload_dir, model_repo_id, 'tokenizer.json'),
            )

            meta_info = {
                'name': model_repo_id.split('/')[-1],
                'repo_id': model_repo_id,
                'image_flops': img_flops,
                'image_params': img_params,
                'image_size': input_img_size,
                'image_encoding_width': img_encoding_width,
                'image_embedding_width': img_embedding_width,
                'text_flops': text_flops,
                'text_params': text_params,
                'text_encoding_width': text_encoding_width,
                'text_embedding_width': text_embedding_width,
                'repo_created_at': repo_created_at,
                'logit_scale': model_raw.logit_scale.detach().numpy().item(),
                'logit_bias': model_raw.logit_bias.detach().numpy().item(),
            }
            with open(os.path.join(upload_dir, model_repo_id, 'meta.json'), 'w') as f:
                json.dump(meta_info, f, sort_keys=True, indent=4)

            c_meta_info = copy.deepcopy(meta_info)
            d_models[meta_info['name']] = c_meta_info

            df_models = pd.DataFrame(list(d_models.values()))
            df_models = df_models.sort_values(by=['repo_created_at'], ascending=False)
            df_models.to_parquet(os.path.join(upload_dir, 'models.parquet'), index=False)

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('pipeline_tag: zero-shot-classification', file=f)
                print('base_model:', file=f)
                for rid in natsorted(set(df_models['repo_id'][:100])):
                    print(f'- {rid}', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('tags:', file=f)
                print('- transformers', file=f)
                print('- siglip', file=f)
                print('- image', file=f)
                print('- dghs-realutils', file=f)
                print('library_name: dghs-realutils', file=f)
                print('---', file=f)
                print('', file=f)

                print('ONNX exported version of SigLIP models.', file=f)
                print('', file=f)

                print(f'# Models', file=f)
                print(f'', file=f)

                df_shown = pd.DataFrame([
                    {
                        "Name": f'[{item["repo_id"]}]({hf_hub_repo_url(repo_id=item["repo_id"], repo_type="model")})',
                        'Image (Params/FLOPS)': f'{clever_format(item["image_params"], "%.1f")} / {clever_format(item["image_flops"], "%.1f")}',
                        'Image Size': item['image_size'],
                        "Image Width (Enc/Emb)": f'{item["image_encoding_width"]} / {item["image_embedding_width"]}',
                        'Text (Params/FLOPS)': f'{clever_format(item["text_params"], "%.1f")} / {clever_format(item["text_flops"], "%.1f")}',
                        "Text Width (Enc/Emb)": f'{item["text_encoding_width"]} / {item["text_embedding_width"]}',
                        'Created At': datetime.datetime.fromtimestamp(item['repo_created_at']).strftime('%Y-%m-%d'),
                        'flops': item['image_flops'],
                        'created_at': item['repo_created_at'],
                    }
                    for item in df_models.to_dict('records')
                ])
                df_shown = df_shown.sort_values(by=['created_at', 'flops'], ascending=[False, False])
                del df_shown['created_at']
                del df_shown['flops']
                print(f'{plural_word(len(df_shown), "model")} exported in total.', file=f)
                print(f'', file=f)
                print(df_shown.to_markdown(index=False), file=f)
                print(f'', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='model',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Export model {model_repo_id!r}',
            )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync()
    # repo_id = "google/siglip-base-patch16-256-multilingual"
    #
    # model_raw = get_siglip_model(repo_id)
    # processor, dummy_input = get_dummy_input(repo_id)
    #
    # # export_text_to_onnx(model_raw, dummy_input)
    # # export_image_to_onnx(model_raw, dummy_input)
    # export_image_preprocessor(processor.image_processor)
    # export_text_tokenizer(repo_id)
