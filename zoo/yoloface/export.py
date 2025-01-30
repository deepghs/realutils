import json
import os.path
from contextlib import contextmanager

import torch
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client
from hfutils.utils import TemporaryDirectory, download_file
from huggingface_hub import CommitOperationAdd
from ultralytics import YOLO

from ..yolo.onnx import export_yolo_to_onnx


@contextmanager
def load_pt_file(model_name: str):
    with TemporaryDirectory() as td:
        dst_filename = os.path.join(td, f'{model_name}.pt')
        url = f'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/{model_name}.pt'
        logging.info(f'Download from {url!r} to {dst_filename!r} ...')
        download_file(
            url,
            dst_filename,
        )
        yield dst_filename


def export(model_name: str, repository: str = 'deepghs/yolo-face', opset_version: int = 14):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=False)

    with load_pt_file(model_name) as model_file:
        model = YOLO(model_file)

        files = []

        with TemporaryDirectory() as workdir:
            best_pt = model_file

            best_pt_exp = os.path.join(workdir, 'model.pt')
            logging.info(f'Copying best pt {best_pt!r} to {best_pt_exp!r}')
            state_dict = torch.load(best_pt)
            if state_dict['train_args']['data']:
                state_dict['train_args']['data'] = sha3(state_dict['train_args']['data'].encode(), n=224)
            if state_dict['train_args']['project']:
                state_dict['train_args']['project'] = sha3(state_dict['train_args']['project'].encode(), n=224)
            if state_dict['train_args']['model'] and \
                    ('/' in state_dict['train_args']['model'] or '\\' in state_dict['train_args']['model']):
                state_dict['train_args']['model'] = sha3(state_dict['train_args']['model'].encode(), n=224)
            torch.save(state_dict, best_pt_exp)
            # shutil.copy(best_pt, best_pt_exp)
            files.append((best_pt_exp, 'model.pt'))

            names_map = YOLO(best_pt).names
            labels = [names_map[i] for i in range(len(names_map))]
            with open(os.path.join(workdir, 'labels.json'), 'w') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)
            files.append((os.path.join(workdir, 'labels.json'), 'labels.json'))
            with open(os.path.join(workdir, 'model_type.json'), 'w') as f:
                json.dump({'model_type': 'yolo'}, f, ensure_ascii=False, indent=4)
            files.append((os.path.join(workdir, 'model_type.json'), 'model_type.json'))

            best_onnx_exp = os.path.join(workdir, f'model.onnx')
            logging.info(f'Export onnx model to {best_onnx_exp!r}')
            export_yolo_to_onnx(model, best_onnx_exp, opset_version=opset_version)
            files.append((best_onnx_exp, 'model.onnx'))

            operations = []
            for src_file, filename in files:
                operations.append(CommitOperationAdd(
                    path_in_repo=f'{model_name}/{filename}',
                    path_or_fileobj=src_file,
                ))

            hf_client.create_commit(
                repo_id=repository,
                repo_type='model',
                operations=operations,
                commit_message=f'Add model {model_name!r}'
            )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    for mn in [
        "yolov6m-face",
        "yolov6n-face",
        "yolov8l-face",
        "yolov8m-face",
        "yolov8n-face",
        "yolov9-c-face",

        "yolov10l-face",
        "yolov10m-face",
        "yolov10n-face",
        "yolov10s-face",

        "yolov11l-face",
        "yolov11m-face",
        "yolov11n-face",
        "yolov11s-face",
    ]:
        export(mn)
