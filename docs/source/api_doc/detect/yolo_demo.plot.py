import glob
import os

from imgutils.detect.visual import detection_visualize
from imgutils.generic.yolo import _open_models_for_repo_id

from plot import image_plot
from realutils.detect import detect_by_yolo
from realutils.detect.yolo import _REPO_ID, _DEFAULT_MODEL

_MODELS = _open_models_for_repo_id(_REPO_ID).model_names
_, _, _LABELS = _open_models_for_repo_id(_REPO_ID)._open_model(_DEFAULT_MODEL)


def _detect(img, **kwargs):
    return detection_visualize(img, detect_by_yolo(img, **kwargs), _LABELS)


if __name__ == '__main__':
    image_plot(
        *[
            (_detect(file), f'Image #{i}')
            for i, file in enumerate(glob.glob(os.path.join('yolo', '*.jpg')), start=1)
        ],
        columns=2,
        figsize=(9, 9),
        autocensor=False,
    )
