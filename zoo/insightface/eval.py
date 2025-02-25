import glob
import mimetypes
import os
from contextlib import contextmanager

import numpy as np
from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_unpack
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


def make_eval_result(model_name: str, rescale_ratio: float = 7):
    for dsname in _AVAILABLE_DS:
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

        quit()


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    make_eval_result('buffalo_l')
    # with mock_eval_dataset('cfpw') as d:
    #     print(d)
    #     os.system(f'ls -al {d!r}')
