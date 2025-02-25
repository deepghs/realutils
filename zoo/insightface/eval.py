import glob
import mimetypes
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_unpack
from hfutils.operate import upload_directory_as_directory, get_hf_client
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
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


def make_eval_result(repo_id: str = 'deepghs/insightface', model_name: str = 'buffalo_l', rescale_ratio: float = 7):
    hf_client = get_hf_client()
    for dsname in _AVAILABLE_DS:
        if hf_client.file_exists(
                repo_id=repo_id,
                repo_type='model',
                filename=f'{model_name}/{dsname}_sims.csv',
        ):
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

            df_src = pd.read_csv(os.path.join(ds_dir, 'paris.csv'))
            records = []
            for record in tqdm(df_src.to_dict('records'), desc='Calculate values'):
                src_file1 = os.path.join(ds_dir, record['file1'])
                npy_file1 = os.path.splitext(src_file1)[0] + '.npy'
                src_file2 = os.path.join(ds_dir, record['file2'])
                npy_file2 = os.path.splitext(src_file2)[0] + '.npy'
                if not os.path.exists(npy_file1) or not os.path.exists(npy_file2):
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

            with TemporaryDirectory() as upload_dir:
                df.to_csv(os.path.join(upload_dir, f'{dsname}_sims.csv'), index=False)
                upload_directory_as_directory(
                    repo_id=repo_id,
                    repo_type='model',
                    path_in_repo=model_name,
                    local_directory=upload_dir,
                    message=f'Add cosine result of dataset {dsname!r} for model {model_name!r}',
                )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    make_eval_result(model_name='buffalo_l')
    # with mock_eval_dataset('cfpw') as d:
    #     print(d)
    #     os.system(f'ls -al {d!r}')
