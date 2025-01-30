import json
import os
from tempfile import TemporaryDirectory

from ditk import logging
from hfutils.operate import get_hf_fs, upload_directory_as_directory, get_hf_client
from imgutils.preprocess import create_transforms_from_transformers, parse_pillow_transforms
from transformers import AutoImageProcessor

from .onnx import onnx_export
from .profile import model_profile


def export(repository: str, model_name: str):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=False)

    with TemporaryDirectory() as upload_dir:
        onnx_file = os.path.join(upload_dir, 'model.onnx')
        embedding_width = onnx_export(onnx_file, model_name=model_name)
        logging.info(f'Embedding with of onnx model: {embedding_width!r}')

        logging.info(f'Profiling model {model_name!r} ...')
        profile = model_profile(model_name)

        preprocessor_file = os.path.join(upload_dir, 'preprocessor.json')
        processor = AutoImageProcessor.from_pretrained(model_name)
        pillow_trans = create_transforms_from_transformers(processor)
        logging.info('Extracted preprocessor stages:\n'
                     f'{pillow_trans}')
        logging.info(f'Writing to {preprocessor_file!r} ...')
        with open(preprocessor_file, 'w') as f:
            json.dump({
                'stages': parse_pillow_transforms(pillow_trans),
            }, f, indent=4, sort_keys=True)

        with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
            json.dump({
                'name': model_name,
                'width': embedding_width,
                'params': profile['params'],
                'flops': profile['flops'],
            }, f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            local_directory=upload_dir,
            path_in_repo=model_name,
            hf_token=os.environ['HF_TOKEN'],
            message=f'Export dinov2 model {model_name!r}'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    export(
        repository=os.environ.get('REPO', 'deepghs/dinov2_onnx'),
        model_name=os.environ['MODEL'],
    )
