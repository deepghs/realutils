import random

from imgutils.generic.yolo import _open_models_for_repo_id

from benchmark import BaseBenchmark, create_plot_cli
from realutils.detect.yolo import detect_by_yolo, _REPO_ID

_MODELS = _open_models_for_repo_id(_REPO_ID).model_names


class YOLOBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from imgutils.generic.yolo import _open_models_for_repo_id
        _ = _open_models_for_repo_id(_REPO_ID)._open_model(self.model_name)

    def unload(self):
        from imgutils.generic.yolo import _open_models_for_repo_id
        _open_models_for_repo_id.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = detect_by_yolo(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model_name, YOLOBenchmark(model_name))
            for model_name in _MODELS
            if model_name.startswith('yolov9')
        ],
        title='Benchmark for YOLOv9 Models',
        run_times=10,
        try_times=20,
    )()
