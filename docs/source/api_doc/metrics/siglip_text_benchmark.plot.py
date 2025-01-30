import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.metrics.clip import get_siglip_text_embedding


class SigLIPTextBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from realutils.metrics.siglip import _open_text_tokenizer, _open_text_encoder
        _ = _open_text_tokenizer(self.model_name)
        _ = _open_text_encoder(self.model_name)

    def unload(self):
        from realutils.metrics.siglip import _open_text_tokenizer, _open_text_encoder
        _open_text_tokenizer.cache_clear()
        _open_text_encoder.cache_clear()

    def run(self):
        text = random.choice(self.all_texts)
        _ = get_siglip_text_embedding(text, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, SigLIPTextBenchmark(model))
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
        title='Benchmark for SigLIP Text Encoder',
        run_times=10,
        try_times=20,
    )()
