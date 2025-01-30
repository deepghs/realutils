import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.metrics.siglip import get_siglip_image_embedding


class SigLIPImageBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from realutils.metrics.siglip import _open_image_preprocessor, _open_image_encoder
        _ = _open_image_preprocessor(self.model_name)
        _ = _open_image_encoder(self.model_name)

    def unload(self):
        from realutils.metrics.siglip import _open_image_preprocessor, _open_image_encoder
        _open_image_preprocessor.cache_clear()
        _open_image_encoder.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_siglip_image_embedding(image_file, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, SigLIPImageBenchmark(model))
            for model in [
            'google/siglip-so400m-patch14-384',
            # 'google/siglip-so400m-patch14-224',
            'google/siglip-base-patch16-256-multilingual',
            'google/siglip-base-patch16-224',
            # 'google/siglip-so400m-patch16-256-i18n',
            'google/siglip-base-patch16-512',
            'google/siglip-large-patch16-256',
            'google/siglip-base-patch16-384',
            'google/siglip-large-patch16-384',
            'google/siglip-base-patch16-256'
        ]
        ],
        title='Benchmark for SigLIP Image Encoder',
        run_times=10,
        try_times=20,
    )()
