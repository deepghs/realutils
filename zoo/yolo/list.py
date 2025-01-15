import io
import json
import os.path

import click
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_file_url
from hfutils.utils import hf_fs_path, parse_hf_fs_path
from huggingface_hub.hf_api import RepoFile
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_num_params, get_flops

from ..utils import GLOBAL_CONTEXT_SETTINGS, float_pe, markdown_to_df


@click.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, default='deepghs/yolos',
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def list_(repository: str, revision: str = 'main'):
    logging.try_init_root(logging.INFO)
    rows = []
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    d_labels = {}
    d_thresholds = {}
    d_model_types = {}
    for pt_path in tqdm(hf_fs.glob(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='*/model.pt',
            revision=revision,
    ))):
        pt_file = parse_hf_fs_path(pt_path).filename
        name = os.path.dirname(pt_file)

        if hf_fs.exists(f'{repository}/{name}/model_type.json'):
            model_type = json.loads(hf_fs.read_text(f'{repository}/{name}/model_type.json'))['model_type']
        else:
            model_type = 'yolo'
        d_model_types[name] = model_type
        if model_type == 'yolo':
            model_cls = YOLO
        else:
            model_cls = RTDETR

        logging.info(f'Making information for {name!r} ...')
        model = model_cls(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename=pt_file,
            revision=revision,
        ))

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/model.pt'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        names_map = model.names
        labels = [names_map[i] for i in range(len(names_map))]
        metrics = {
            key.split('/', maxsplit=1)[-1]: f'{value:.3g}'
            for key, value in dict(model.ckpt.get('train_metrics') or {}).items()
            if key.startswith('metrics/')
        }
        row = {
            'Model': name,
            'Type': model_type,
            'FLOPS': float_pe(get_flops(model.model) * 1e9),
            'Params': float_pe(get_num_params(model.model)),
        }
        logging.info(f'Model {name!r}: {row!r}')

        row = {**row, **metrics}
        d_labels[name] = labels
        if len(labels) <= 5:
            label_text = ', '.join(map(lambda x: f'`{x}`', labels))
            row['Labels'] = label_text
        else:
            label_text = ', '.join(map(lambda x: f'`{x}`', labels[:5])) + \
                         f' ... {plural_word(len(labels), "label")} in total'
            file_url = hf_hub_repo_file_url(
                repo_id=repository,
                repo_type='model',
                path=f'{name}/labels.json',
                revision=revision,
            )
            row['Labels'] = f'[{label_text}]({file_url})'
        row['created_at'] = last_commit_at
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        for name, labels in d_labels.items():
            os.makedirs(os.path.join(td, name), exist_ok=True)
            with open(os.path.join(td, name, 'labels.json'), 'w') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)
        for name, (max_f1_score, threshold) in d_thresholds.items():
            os.makedirs(os.path.join(td, name), exist_ok=True)
            with open(os.path.join(td, name, 'threshold.json'), 'w') as f:
                json.dump({
                    'f1_score': max_f1_score,
                    'threshold': threshold,
                }, f, ensure_ascii=False, indent=4)
        for name, model_type in d_model_types.items():
            os.makedirs(os.path.join(td, name), exist_ok=True)
            with open(os.path.join(td, name, 'model_type.json'), 'w') as f:
                json.dump({'model_type': model_type}, f, ensure_ascii=False, indent=4)

        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Model' in df_c.columns and 'FLOPS' in df_c.columns and \
                                        'Params' in df_c.columns and 'Labels' in df_c.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                                else:
                                    print(os.linesep.join(tb_lines), file=f)
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    list_()
