import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.metrics.dinov2 import get_dinov2_embedding


class Dinov2Benchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from realutils.metrics.dinov2 import _get_dinov2_model, _get_preprocessor
        _ = _get_dinov2_model(self.model_name)
        _ = _get_preprocessor(self.model_name)

    def unload(self):
        from realutils.metrics.dinov2 import _get_dinov2_model, _get_preprocessor
        _get_dinov2_model.cache_clear()
        _get_preprocessor.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_dinov2_embedding(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, Dinov2Benchmark(model))
            for model in [
                'facebook/dinov2-small',
                'facebook/dinov2-base',
                'facebook/dinov2-large',

                'facebook/dino-vits16',
                'facebook/dino-vitb16',
                'facebook/dino-vits8',
                'facebook/dino-vitb8',
            ]
        ],
        title='Benchmark for Dinov2 Models',
        run_times=10,
        try_times=20,
    )()
