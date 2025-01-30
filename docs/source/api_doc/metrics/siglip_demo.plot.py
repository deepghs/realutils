import glob
import io
import os.path

from natsort import natsorted
from tqdm import tqdm

from plot import image_plot
from realutils.metrics import classify_with_siglip


def _make_label(file):
    with io.StringIO() as sf:
        print(file, file=sf)
        texts = [
            'a photo of a cat',
            'a photo of 2 cats',
            'a photo of 2 dogs',
            'a photo of a woman',
        ]
        preds = classify_with_siglip(images=[file], texts=texts)[0]
        for text, score in zip(texts, preds.tolist()):
            print(f'{text}: {score:.4f}', file=sf)

        return sf.getvalue()


if __name__ == '__main__':
    image_plot(
        *[
            (file, _make_label(file))
            for file in tqdm(natsorted(glob.glob(os.path.join('xlip', '*.jpg'))))
        ],
        columns=2,
        figsize=(8, 16),
    )
