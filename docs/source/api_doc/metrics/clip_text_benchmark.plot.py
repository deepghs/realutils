import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.metrics.clip import get_clip_text_embedding


class CLIPTextBenchmark(BaseBenchmark):
    def __init__(self, model_name: str):
        BaseBenchmark.__init__(self)
        self.model_name = model_name

    def load(self):
        from realutils.metrics.clip import _open_text_tokenizer, _open_text_encoder
        _ = _open_text_tokenizer(self.model_name)
        _ = _open_text_encoder(self.model_name)

    def unload(self):
        from realutils.metrics.clip import _open_text_tokenizer, _open_text_encoder
        _open_text_tokenizer.cache_clear()
        _open_text_encoder.cache_clear()

    def run(self):
        text = random.choice(self.all_texts)
        _ = get_clip_text_embedding(text, model_name=self.model_name)


if __name__ == '__main__':
    create_plot_cli(
        [
            (model, CLIPTextBenchmark(model))
            for model in [
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-base-patch16',
            'openai/clip-vit-large-patch14',
            'openai/clip-vit-large-patch14-336',
        ]
        ],
        title='Benchmark for CLIP Text Encoder',
        run_times=10,
        try_times=20,
    )()
