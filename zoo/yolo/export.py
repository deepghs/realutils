import json
import os.path
from typing import Union, Optional

import torch
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client
from hfutils.utils import TemporaryDirectory
from huggingface_hub import CommitOperationAdd
from ultralytics import YOLO, RTDETR

from .onnx import export_yolo_to_onnx


def export(repository: str = 'deepghs/yolos',
           level: str = 's', yversion: Union[int, str] = 8, opset_version: int = 14,
           model_name: Optional[str] = None):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', private=False)

    if not model_name:
        if yversion == 11 or yversion == '11':
            model_file = f'yolo11{level}.pt'
            model = YOLO(model_file)
            model_type = 'yolo'
        elif isinstance(yversion, str) and yversion.lower() == 'rtdetr':
            model_file = f'rtdetr-{level}.pt'
            model = RTDETR(model_file)
            model_type = 'rtdetr'
        else:
            model_file = f'yolov{yversion}{level}.pt'
            model = YOLO(model_file)
            model_type = 'yolo'
        model_name, _ = os.path.splitext(model_file)
    else:
        model_file = f'{model_name}.pt'
        model = YOLO(model_file)
        model_type = 'yolo'

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
                '/' in state_dict['train_args']['model'] or '\\' in state_dict['train_args']['model']:
            state_dict['train_args']['model'] = sha3(state_dict['train_args']['model'].encode(), n=224)
        torch.save(state_dict, best_pt_exp)
        # shutil.copy(best_pt, best_pt_exp)
        files.append((best_pt_exp, 'model.pt'))

        if model_type == 'yolo':
            names_map = YOLO(best_pt).names
        else:
            names_map = RTDETR(best_pt).names
        labels = [names_map[i] for i in range(len(names_map))]
        with open(os.path.join(workdir, 'labels.json'), 'w') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        files.append((os.path.join(workdir, 'labels.json'), 'labels.json'))
        with open(os.path.join(workdir, 'model_type.json'), 'w') as f:
            json.dump({'model_type': model_type}, f, ensure_ascii=False, indent=4)
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
    # export(model_name='yolov5nu')
    # export(model_name='yolov5su')
    # export(model_name='yolov5mu')
    # export(model_name='yolov5lu')
    # export(model_name='yolov5xu')
    #
    # export(model_name='yolov8n')
    # export(model_name='yolov8s')
    # export(model_name='yolov8m')
    # export(model_name='yolov8l')
    # export(model_name='yolov8x')
    #
    # export(model_name='yolov9t')
    # export(model_name='yolov9s')
    # export(model_name='yolov9m')
    # export(model_name='yolov9c')
    # export(model_name='yolov9e')

    export(model_name='yolov10n')
    export(model_name='yolov10s')
    export(model_name='yolov10m')
    export(model_name='yolov10b')
    export(model_name='yolov10l')
    export(model_name='yolov10x')

    export(model_name='yolo11n')
    export(model_name='yolo11s')
    export(model_name='yolo11m')
    export(model_name='yolo11l')
    export(model_name='yolo11x')

    export(model_name='rtdetr-l.pt')
    export(model_name='rtdetr-x.pt')
