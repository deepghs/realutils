import glob
import json
import mimetypes
import os
from contextlib import contextmanager
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_unpack
from hfutils.operate import upload_directory_as_directory, get_hf_client
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_curve
from tqdm import tqdm

from realutils.face.insightface import isf_detect_faces, isf_extract_face

_AVAILABLE_DS = [
    'CFPW',
    'LFW',
    'CALFW',
    'CPLFW',
]


@contextmanager
def mock_eval_dataset(dsname: str):
    with TemporaryDirectory() as td:
        zip_file = hf_hub_download(
            repo_id='deepghs/face_eval_pairs',
            repo_type='dataset',
            filename=f'{dsname.lower()}.zip',
        )
        archive_unpack(
            archive_file=zip_file,
            directory=td,
        )

        yield td


def plot_by_df(df: pd.DataFrame, title: str):
    y_true = np.where(df['pair_type'] == 'same', 1, 0)
    y_scores = df['cos_sim'].values

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision_pr, recall_pr, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_pr, precision_pr)

    f1_scores = 2 * (precision_pr[:-1] * recall_pr[:-1]) / (precision_pr[:-1] + recall_pr[:-1] + 1e-8)
    max_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]
    optimal_threshold = pr_thresholds[max_f1_idx]

    thresholds = np.linspace(0, 1, 1000)
    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    plt.cla()
    plt.figure(figsize=(12, 9))
    plt.suptitle(title)

    plt.subplot(2, 3, 1)
    plt.plot(thresholds, precisions)
    plt.title('Threshold vs Precision')
    plt.xlabel('Threshold'), plt.ylabel('Precision')
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(thresholds, recalls)
    plt.title('Threshold vs Recall')
    plt.xlabel('Threshold'), plt.ylabel('Recall')
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(recall_pr, precision_pr)
    plt.title(f'PR Curve (AUC={pr_auc:.3f})')
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(thresholds, f1s)
    plt.axvline(optimal_threshold, color='r', linestyle='--')
    plt.title(f'Threshold vs F1\n'
              f'(Max F1={max_f1:.3f}, Threshold={optimal_threshold:.3f})')
    plt.xlabel('Threshold'), plt.ylabel('F1 Score')
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve (AUC={roc_auc:.3f})')
    plt.xlabel('FPR'), plt.ylabel('TPR')
    plt.grid()

    plt.tight_layout()
    with TemporaryDirectory() as td:
        image_file = os.path.join(td, 'image.png')
        plt.savefig(image_file, dpi=200)
        image = Image.open(image_file)
        image.load()
    plt.cla()

    metrics = {
        'max_f1': max_f1,
        'optimal_threshold': optimal_threshold,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }
    return metrics, image


def make_eval_result(repo_id: str = 'deepghs/insightface', model_name: str = 'buffalo_l', rescale_ratio: float = 7):
    hf_client = get_hf_client()
    global_records = []
    for dsname in _AVAILABLE_DS:
        if hf_client.file_exists(
                repo_id=repo_id,
                repo_type='model',
                filename=f'{model_name}/{dsname}_sims.csv',
        ):
            global_records.extend([
                {**item, 'source': dsname}
                for item in pd.read_csv(hf_client.hf_hub_download(
                    repo_id=repo_id,
                    repo_type='model',
                    filename=f'{model_name}/{dsname}_sims.csv',
                )).to_dict('records')
            ])
            logging.warn(f'Result for {dsname!r} already exist, skipped.')
            continue

        logging.info(f'Eval {model_name!r} with dataset {dsname!r} ...')
        with mock_eval_dataset(dsname) as ds_dir:
            for image_file in tqdm(glob.glob(os.path.join(ds_dir, 'images', '**', '*'), recursive=True),
                                   desc=f'Embeddings for {dsname!r}'):
                mimetype, _ = mimetypes.guess_type(image_file)
                if not mimetype.startswith('image/'):
                    continue

                image = load_image(image_file, force_background='white', mode='RGB')
                padded_image = Image.new('RGB', (int(image.width * rescale_ratio),
                                                 int(image.height * rescale_ratio)), 'white')
                paste_x = int(image.width * rescale_ratio - image.width) // 2
                paste_y = int(image.height * rescale_ratio - image.height) // 2
                padded_image.paste(image, (paste_x, paste_y))
                image = padded_image

                faces = isf_detect_faces(image, model_name=model_name)
                mx, my = image.width / 2, image.height / 2
                if not faces:
                    logging.info(f'No face detected in {image_file!r}, skipped.')
                    continue

                face_orders = []
                for fi, face in enumerate(faces):
                    cx, cy = np.array(face.keypoints).mean(axis=0).tolist()
                    dist = ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5
                    face_orders.append((dist, fi, face))
                face_orders = sorted(face_orders)
                _, _, face = face_orders[0]

                embedding = isf_extract_face(image, face, model_name=model_name)
                dst_emb_file = os.path.splitext(image_file)[0] + '.npy'
                np.save(dst_emb_file, embedding)

            df_src = pd.read_csv(os.path.join(ds_dir, 'pairs.csv'))
            records = []
            for record in tqdm(df_src.to_dict('records'), desc='Calculate values'):
                src_file1 = os.path.join(ds_dir, record['file1'])
                npy_file1 = os.path.splitext(src_file1)[0] + '.npy'
                src_file2 = os.path.join(ds_dir, record['file2'])
                npy_file2 = os.path.splitext(src_file2)[0] + '.npy'
                if not os.path.exists(npy_file1) or not os.path.exists(npy_file2):
                    logging.warning(f'One of the file in {record!r} not exist, skipped.')
                    continue

                emb1, emb2 = np.load(npy_file1), np.load(npy_file2)
                emb1 = emb1 / np.linalg.norm(emb1)
                emb2 = emb2 / np.linalg.norm(emb2)
                cosine = (emb1 * emb2).sum().item()
                records.append({
                    **record,
                    'cos_sim': cosine,
                })

            df = pd.DataFrame(records)
            logging.info(f'Results:\n{df}')
            global_records.extend([{**item, 'source': dsname} for item in df.to_dict('records')])

            metrics, plt_image = plot_by_df(df, title=f'Eval result of model {model_name!r} on dataset {dsname!r}')
            logging.info(f'Metrics:\n{pformat(metrics)}')

            with TemporaryDirectory() as upload_dir:
                dst_csv_file = os.path.join(upload_dir, f'{dsname}_sims.csv')
                os.makedirs(os.path.dirname(dst_csv_file), exist_ok=True)
                df.to_csv(dst_csv_file, index=False)

                with open(os.path.join(upload_dir, f'{dsname}_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=4, sort_keys=True)
                plt_image.save(os.path.join(upload_dir, f'{dsname}_plot.png'))

                upload_directory_as_directory(
                    repo_id=repo_id,
                    repo_type='model',
                    path_in_repo=model_name,
                    local_directory=upload_dir,
                    message=f'Add cosine result of dataset {dsname!r} for model {model_name!r}',
                )

    df = pd.DataFrame(global_records)
    metrics, plt_image = plot_by_df(df, title=f'Eval result of model {model_name!r} on ALL datasets')
    logging.info(f'Global Metrics:\n{pformat(metrics)}')

    with TemporaryDirectory() as upload_dir:
        dst_csv_file = os.path.join(upload_dir, f'sims.csv')
        os.makedirs(os.path.dirname(dst_csv_file), exist_ok=True)
        df.to_csv(dst_csv_file, index=False)

        with open(os.path.join(upload_dir, f'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        plt_image.save(os.path.join(upload_dir, f'plot.png'))

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            path_in_repo=model_name,
            local_directory=upload_dir,
            message=f'Add cosine result of ALL dataset for model {model_name!r}',
        )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    make_eval_result(model_name='buffalo_s')
    # with mock_eval_dataset('cfpw') as d:
    #     print(d)
    #     os.system(f'ls -al {d!r}')
