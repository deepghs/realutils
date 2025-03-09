import io
import json
import os

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import parse_hf_fs_path, hf_normpath, hf_fs_path
from huggingface_hub.hf_api import RepoFile
from thop import clever_format

from ..utils import markdown_to_df


def list_(repository: str):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    rows = []
    for model_file in hf_fs.glob(f'{repository}/**/model.onnx'):
        model_name = hf_normpath(os.path.dirname(parse_hf_fs_path(model_file).filename))
        logging.info(f'Processing model {model_name!r} ...')

        meta_info = json.loads(hf_fs.read_text(f'{repository}/{model_name}/meta.json'))
        params, flops = clever_format([meta_info['params'], meta_info['flops']], "%.1f")

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{model_name}/model.onnx'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        rows.append({
            'Model': f'[{model_name}]({hf_hub_repo_url(repo_id=repository, repo_type="model")})',
            'Params': params,
            'FLOPS': flops,
            'Width': meta_info['width'],
            'created_at': last_commit_at,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Model' in df_c.columns and 'FLOPS' in df_c.columns and \
                                        'Params' in df_c.columns and 'Width' in df_c.columns:
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
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    list_(
        repository=os.environ.get('REPO', 'deepghs/dinov2_onnx'),
    )
