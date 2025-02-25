import json
import os.path

import pandas as pd
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import parse_hf_fs_path
from tqdm import tqdm


def sync(repo_id: str):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    model_names = [
        os.path.dirname(parse_hf_fs_path(path).filename)
        for path in hf_fs.glob(f'{repo_id}/*/metrics.json')
    ]
    logging.info(f'Available model names: {model_names!r}')

    records = []
    for model_name in tqdm(model_names):
        metrics = json.loads(hf_fs.read_text(f'{repo_id}/{model_name}/metrics.json'))

        def _fn(x):
            return f'{x["det_ratio"] * 100.0:.2f}% / ' \
                   f'{x["max_f1"] * 100.0:.2f}% / ' \
                   f'{x["optimal_threshold"]:.4f}'

        row = {
            'Model': model_name,
            'Eval ALL (Det/Rec-F1/Rec-Thresh)': _fn(metrics),
        }
        for dsname, dsmetrics in metrics['datasets'].items():
            row[f'Eval {dsname} (Det/Rec-F1/Rec-Thresh)'] = _fn(dsmetrics)
        records.append(row)

    with TemporaryDirectory() as upload_dir:
        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('license_name: model-distribution-disclaimer-license', file=f)
            print('license_link: https://huggingface.co/spaces/deepghs/RDLicence', file=f)
            print('pipeline_tag: feature-extraction', file=f)
            print('tags:', file=f)
            print('- onnx', file=f)
            print('- face', file=f)
            print('---', file=f)
            print('', file=f)

            print('ONNX models from [insightface project](https://github.com/deepinsight/insightface).', file=f)
            print('', file=f)

            print(f'# Available Models', file=f)
            print(f'', file=f)

            print(f'We evaluated all these models with some evaluation datasets on face recognition.', file=f)
            print(f'', file=f)
            print('* CFPW (500 ids/7K images/7K pairs)[1]', file=f)
            print('* LFW (5749 ids/13233 images/6K pairs)[2]', file=f)
            print('* CALFW (5749 ids/13233 images/6K pairs)[3]', file=f)
            print('* CPLFW (5749 ids/13233 images/6K pairs)[4]', file=f)
            print(f'', file=f)

            print(f'Below are the complete results and recommended thresholds.', file=f)
            print(f'', file=f)
            print(f'* Det: Success rate of face detection and landmark localization.', file=f)
            print(f'* Rec-F1: Maximum F1 score achieved in face recognition.', file=f)
            print(f'* Rec-Thresh: Optimal threshold determined by the maximum F1 score.', file=f)
            print(f'', file=f)

            df = pd.DataFrame(records)
            print(df.to_markdown(index=False), file=f)
            print(f'', file=f)

            print(
                '[1] Sengupta Soumyadip, Chen Jun-Cheng, Castillo Carlos, Patel Vishal M, Chellappa Rama, Jacobs David W, Frontal to profile face verification in the wild, WACV, 2016.',
                file=f)
            print(f'', file=f)
            print(
                '[2] Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments, 2007.',
                file=f)
            print(f'', file=f)
            print(
                '[3] Zheng Tianyue, Deng Weihong, Hu Jiani, Cross-age lfw: A database for studying cross-age face recognition in unconstrained environments, arXiv:1708.08197, 2017.',
                file=f)
            print(f'', file=f)
            print(
                '[4] Zheng, Tianyue, and Weihong Deng. Cross-Pose LFW: A Database for Studying Cross-Pose Face Recognition in Unconstrained Environments, 2018.',
                file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            path_in_repo='.',
            local_directory=upload_dir,
            message=f'List ALL the models',
        )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repo_id='deepghs/insightface',
    )
