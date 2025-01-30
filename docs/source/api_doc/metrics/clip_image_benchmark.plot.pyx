import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.metrics.clip import get_clip_image_embedding


class CLIPImageBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from realutils.metrics.clip import _open_image_preprocessor, _open_image_encoder
        _ = _open_image_preprocessor(self.model_name)
        _ = _open_image_encoder(self.model_name)

    def unload(self):
        from realutils.metrics.clip import _open_image_preprocessor, _open_image_encoder
        _open_image_preprocessor.cache_clear()
        _open_image_encoder.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_clip_image_embedding(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, CLIPImageBenchmark(model))
            for model in [
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-base-patch16',
            'openai/clip-vit-large-patch14',
            'openai/clip-vit-large-patch14-336',
        ]
        ],
        title='Benchmark for CLIP Image Encoder',
        run_times=10,
        try_times=20,
    )()
