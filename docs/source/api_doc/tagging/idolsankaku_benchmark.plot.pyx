import random

from benchmark import BaseBenchmark, create_plot_cli
from realutils.tagging import get_idolsankaku_tags


class IdolSankakuBenchmark(BaseBenchmark):
    def __init__(self, model):
        BaseBenchmark.__init__(self)
        self.model = model

    def load(self):
        from realutils.tagging.idolsankaku import _get_idolsankaku_model, _get_idolsankaku_labels
        _ = _get_idolsankaku_model(self.model)
        _ = _get_idolsankaku_labels(self.model)

    def unload(self):
        from realutils.tagging.idolsankaku import _get_idolsankaku_model, _get_idolsankaku_labels
        _get_idolsankaku_model.cache_clear()
        _get_idolsankaku_labels.cache_clear()

    def run(self):
        image_file = random.choice(self.all_images)
        _ = get_idolsankaku_tags(image_file, model_name=self.model)


if __name__ == '__main__':
    create_plot_cli(
        [
            ('idolsankaku-swinv2-tagger-v1', IdolSankakuBenchmark("SwinV2")),
            ('idolsankaku-eva02-large-tagger-v1', IdolSankakuBenchmark("EVA02_Large")),
        ],
        title='Benchmark for Tagging Models',
        run_times=10,
        try_times=20,
    )()
