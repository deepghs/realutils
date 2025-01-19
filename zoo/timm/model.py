import copy
import datetime
import json
import os.path
import random
from typing import Optional

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
from PIL import Image
from ditk import logging
from hbutils.random import global_seed
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_cache
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from natsort import natsorted
from thop import profile, clever_format
from timm import create_model, list_pretrained
from timm.data import resolve_model_data_config, create_transform, infer_imagenet_subset, ImageNetInfo
from torch import nn
from tqdm import tqdm

from .preprocess import export_trans
from ..testings import get_testfile
from ..utils import onnx_optimize


def create_timm_model(repo_id: str, pretrained: bool = False, seed: Optional[int] = 0):
    if seed is not None:
        logging.info(f'Set global seed to {seed!r}.')
        global_seed(seed)

    logging.info(f'Create model from repository {repo_id!r}, pretrained: {pretrained!r}.')
    model = create_model(f'hf-hub:{repo_id}', pretrained=pretrained)
    model.eval()

    data_config = resolve_model_data_config(model)
    transforms = create_transform(**data_config, is_training=False)
    logging.info(f'Transforms of model {type(model)!r}:\n{transforms}')
    return model, transforms


class ModuleWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, classifier: nn.Module):
        super().__init__()
        self.base_module = base_module
        self.classifier = classifier

        self._output_features = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input_tensor, output_tensor):
            assert isinstance(input_tensor, tuple) and len(input_tensor) == 1
            input_tensor = input_tensor[0]
            self._output_features = input_tensor

        self.classifier.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor):
        logits = self.base_module(x)
        preds = torch.softmax(logits, dim=-1)

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features, logits, preds


def extract(export_dir: str, model_repo_id: str, pretrained: bool = True, seed: Optional[int] = 0,
            no_optimize: bool = False):
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    repo_created_at = hf_client.repo_info(repo_id=model_repo_id, repo_type='model').created_at.timestamp()

    model, transforms = create_timm_model(model_repo_id, pretrained, seed)
    image = Image.open(get_testfile('sample_img_1.jpg'))
    dummy_input = transforms(image).unsqueeze(0)
    logging.info(f'Dummy input size: {dummy_input.shape!r}')

    with torch.no_grad():
        expected_dummy_output = model(dummy_input)
    logging.info(f'Dummy output size: {expected_dummy_output.shape!r}')

    classifier = model.get_classifier()
    classifier_position = None
    for name, module in model.named_modules():
        if module is classifier:
            classifier_position = name
            break
    if not classifier_position:
        raise RuntimeError(f'No classifier module found in model {type(model)}.')
    logging.info(f'Classifier module found at {classifier_position!r}:\n{classifier}')

    wrapped_model = ModuleWrapper(model, classifier=classifier)
    with torch.no_grad():
        conv_features, conv_output, conv_preds = wrapped_model(dummy_input)
    logging.info(f'Shape of embeddings: {conv_features.shape!r}')
    logging.info(f'Sample of expected logits:\n'
                 f'{expected_dummy_output[:, -10:]}\n'
                 f'Sample of actual logits:\n'
                 f'{conv_output[:, -10:]}')
    close_matrix = torch.isclose(expected_dummy_output, conv_output, atol=1e-3)
    ratio = close_matrix.type(torch.float32).mean()
    logging.info(f'{ratio * 100:.2f}% of the logits value are the same.')
    assert close_matrix.all(), 'Not all values can match.'

    logging.info('Profiling model ...')
    macs, params = profile(model, inputs=(dummy_input,))
    s_macs, s_params = clever_format([macs, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_macs}')

    config_info = json.loads(hf_fs.read_text(f'{model_repo_id}/config.json'))
    if config_info['num_classes'] > 0:
        classify_supported = True
        if 'label_names' in config_info:
            label_names = config_info['label_names']
            dataset_info = None
            dataset_name = 'custom'
        else:
            subset = infer_imagenet_subset(model)
            dataset_info = ImageNetInfo(subset)
            label_names = dataset_info.label_names()
            dataset_name = subset

        if 'label_descriptions' in config_info:
            label_descs = [config_info['label_descriptions'][name] for name in label_names]
        else:
            if dataset_info:
                label_descs = [dataset_info._lemmas[name] for name in label_names]
            else:
                label_descs = None
        if dataset_info:
            label_definitions = [dataset_info._definitions[name] for name in label_names]
        else:
            label_definitions = None

        all_labels = {'names': label_names}
        if label_descs:
            all_labels['descriptions'] = label_descs
        if label_definitions:
            all_labels['definitions'] = label_definitions
    else:
        classify_supported = False
        label_names = None
        all_labels = None
        dataset_name = None

    with open(os.path.join(export_dir, 'meta.json'), 'w') as f:
        json.dump({
            'num_classes': 0 if not classifier_position else conv_preds.shape[-1],
            'num_features': conv_features.shape[-1],
            'params': params,
            'flops': macs,
            'classify_supported': classify_supported,
            'labels': label_names,
            'other_labels': all_labels,
            'dataset': dataset_name,
            'architecture': config_info['architecture'],
            'pretrained': pretrained,
            'repo_id': model_repo_id,
            'name': model_repo_id.split('/')[-1],
            'repo_created_at': repo_created_at,
            'model_cls': type(model).__name__,
            'input_size': dummy_input.shape[2],
        }, f, indent=4, sort_keys=True)

    with open(os.path.join(export_dir, 'preprocess.json'), 'w') as f:
        json.dump({
            'stages': export_trans(transforms),
        }, f, indent=4, sort_keys=True)

    onnx_filename = os.path.join(export_dir, 'model.onnx')
    with TemporaryDirectory() as td:
        temp_model_onnx = os.path.join(td, 'model.onnx')
        logging.info(f'Exporting temporary ONNX model to {temp_model_onnx!r} ...')
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            temp_model_onnx,
            input_names=['input'],
            output_names=['embedding', 'logits', 'output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            custom_opsets=None,
        )

        model = onnx.load(temp_model_onnx)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        logging.info(f'Complete model saving to {onnx_filename!r} ...')
        onnx.save(model, onnx_filename)

        session = onnxruntime.InferenceSession(onnx_filename)
        o_logits, o_embeddings = session.run(['logits', 'embedding'], {'input': dummy_input.numpy()})
        emb_1 = o_embeddings / np.linalg.norm(o_embeddings, axis=-1, keepdims=True)
        emb_2 = conv_features.numpy() / np.linalg.norm(conv_features.numpy(), axis=-1, keepdims=True)
        emb_sims = (emb_1 * emb_2).sum()
        logging.info(f'Similarity of the embeddings is {emb_sims:.5f}.')
        assert emb_sims >= 0.98, f'Similarity of the embeddings is {emb_sims:.5f}, ONNX validation failed.'


def sync(repository: str = 'deepghs/timms', max_count: int = 100, params_limit: float = 0.5 * 1000 ** 3):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=True)
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

    all_pretrained = list_pretrained()
    random.shuffle(all_pretrained)
    exported_count = 0
    for model_name in tqdm(all_pretrained, desc='Exporting Models'):
        if model_name.startswith('tf_'):
            continue

        model_repo_id = f'timm/{model_name}'
        if not hf_client.repo_exists(repo_id=model_repo_id, repo_type='model'):
            logging.warn(f'Repo {model_repo_id!r} not exist, skipped.')
            continue

        if hf_client.file_exists(
                repo_id=repository,
                repo_type='model',
                filename=f'{model_name}/model.onnx',
        ):
            logging.warn(f'Model {model_name!r} already exported, skipped.')
            continue

        logging.info(f'Checking {model_repo_id!r} ...')
        repo_info = hf_client.repo_info(repo_id=model_repo_id, repo_type='model')
        if repo_info.safetensors and repo_info.safetensors.total:
            params = repo_info.safetensors.total
        elif hf_client.file_exists(
                repo_id=model_repo_id,
                repo_type='model',
                filename='model.safetensors',
        ):
            params = hf_fs.size(f'{model_repo_id}/model.safetensors') // 4
        elif hf_client.file_exists(
                repo_id=model_repo_id,
                repo_type='model',
                filename='pytorch_model.bin',
        ):
            params = hf_fs.size(f'{model_repo_id}/pytorch_model.bin') // 4
        else:
            logging.warn(f'No size or model file found for {model_repo_id!r}, skipped.')
            continue

        logging.info(f'Params of model {model_repo_id!r} is approx {(params / 1000 ** 3):.3g}B.')
        if params > params_limit:
            logging.warn(
                f'Model {model_repo_id!r} is just too large {(params / 1000 ** 3):.3g}B, skipped.')
            continue

        with TemporaryDirectory() as upload_dir:
            logging.info(f'Exporting model {model_name!r} ...')
            os.makedirs(os.path.join(upload_dir, model_name), exist_ok=True)
            try:
                extract(
                    export_dir=os.path.join(upload_dir, model_name),
                    model_repo_id=model_repo_id,
                    pretrained=True,
                    seed=0,
                    no_optimize=False,
                )
            except Exception:
                logging.exception(f'Error when exporting {model_name!r}, skipped.')
                continue

            with open(os.path.join(upload_dir, model_name, 'meta.json'), 'r') as f:
                meta_info = json.load(f)
            c_meta_info = copy.deepcopy(meta_info)
            del c_meta_info['other_labels']
            del c_meta_info['labels']
            d_models[meta_info['name']] = c_meta_info

            df_models = pd.DataFrame(list(d_models.values()))
            df_models = df_models.sort_values(by=['repo_created_at'], ascending=False)
            df_models.to_parquet(os.path.join(upload_dir, 'models.parquet'), index=False)

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('pipeline_tag: image-classification', file=f)
                print('base_model:', file=f)
                for rid in natsorted(set(df_models['repo_id'])):
                    print(f'- {rid}', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('tags:', file=f)
                print('- timm', file=f)
                print('- image', file=f)
                print('- dghs-realutils', file=f)
                print('library_name: dghs-imgutils', file=f)
                print('---', file=f)
                print('', file=f)

                print('ONNX export version from [TIMM](https://huggingface.co/timm).', file=f)
                print('', file=f)

                print(f'# Models', file=f)
                print(f'', file=f)

                df_shown = pd.DataFrame([
                    {
                        "Name": f'[{item["name"]}]({hf_hub_repo_url(repo_id=item["repo_id"], repo_type="model")})',
                        'Params': clever_format(item["params"], "%.1f"),
                        'Flops': clever_format(item["flops"], "%.1f"),
                        'Input Size': item['input_size'],
                        'Can Classify': item['classify_supported'],
                        "Features": item['num_features'],
                        "Classes": item['num_classes'],
                        "Dataset": item['dataset'],
                        'Model': item['model_cls'],
                        'Architecture': item['architecture'],
                        'Created At': datetime.datetime.fromtimestamp(item['repo_created_at']).strftime('%Y-%m-%d'),
                        'flops': item['flops'],
                        'created_at': item['repo_created_at'],
                    }
                    for item in df_models.to_dict('records')
                ])
                print(f'{plural_word(len(df_shown), "model")} exported from TIMM in total.', file=f)
                print(f'', file=f)

                for m_name in natsorted(set(df_shown['Model'])):
                    print(f'{m_name}', file=f)
                    print(f'', file=f)
                    df_shown_m = df_shown[df_shown['Model'] == m_name]
                    df_shown_m = df_shown_m.sort_values(by=['flops', 'created_at'], ascending=[False, False])
                    del df_shown_m['flops']
                    del df_shown_m['created_at']
                    print(f'{plural_word(len(df_shown_m), "model")} with model class `{m_name}`.', file=f)
                    print(f'', file=f)
                    print(df_shown_m.to_markdown(index=False), file=f)
                    print(f'', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='model',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Export model {model_name!r}',
            )

            exported_count += 1
            if exported_count >= max_count:
                break


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repository='deepghs/timms',
        params_limit=0.1 * 1000 ** 3,
        max_count=100,
    )
    # repo_id = 'timm/mobilenetv3_large_150d.ra4_e3600_r256_in1k'
    # repo_id = 'timm/eva02_large_patch14_clip_336.merged2b'
    # repo_id = 'timm/eva02_large_patch14_clip_336.merged2b_ft_inat21'
    # repo_id = 'timm/convnext_nano.r384_ad_in12k'
    # repo_id = 'timm/mobilevitv2_200.cvnets_in22k_ft_in1k_384'
    # repo_id = 'timm/caformer_s18.sail_in22k_ft_in1k_384'
    # repo_id = 'timm/caformer_m36.sail_in22k'
    # repo_id = 'timm/resnet50x4_clip_gap.openai'
    # repo_id = 'timm/resnetv2_34d.ra4_e3600_r384_in1k'
    # extract(
    #     model_repo_id=repo_id,
    #     pretrained=False,
    #     export_dir='.'
    # )
