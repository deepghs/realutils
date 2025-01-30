import copy
import glob
import multiprocessing
import os
import time
import warnings
from multiprocessing import Process
from typing import Tuple, List

import click
import matplotlib.pyplot as plt
import numpy as np
import psutil
from hbutils.scale import size_to_bytes_str
from hbutils.string import ordinalize, plural_word
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm

from conf import PROJ_DIR
from plot import INCHES_TO_PIXELS

_DEFAULT_IMAGE_POOL = glob.glob(os.path.join(PROJ_DIR, 'test', 'testfile', 'dataset', '**', '*.jpg'), recursive=True)
_DEFAULT_TEXT_POOL = [
    "sunset",
    "red apple",
    "happy dogs",
    "flying birds",
    "ocean waves",
    "cat sleeping in sunlight",
    "fresh bread from bakery",
    "children playing with bubbles",
    "snow falling at night",
    "green leaves in spring",
    "a group of friends having coffee together",
    "beautiful flowers blooming in the garden path",
    "old bicycle leaning against brick wall outside",
    "young artist painting landscape in the park",
    "tourists taking photos at famous landmarks today",
    "the ancient castle stands majestically on top of the misty mountain",
    "a small fishing boat gently rocks on the calm blue sea",
    "colorful butterflies flutter around the bright purple wildflowers in meadow",
    "professional chef preparing gourmet dishes in modern restaurant kitchen",
    "street musician playing violin under the warm glow of streetlights",
    "the golden retriever puppy chases its tail while playing in the fresh green grass",
    "morning sunlight streams through stained glass windows of the historic gothic cathedral",
    "experienced rock climber carefully makes her way up the challenging vertical cliff face",
    "talented ballet dancers perform gracefully on stage during the annual winter performance",
    "local farmers selling fresh organic vegetables and fruits at the weekend market",
    "the skilled photographer captures stunning images of wild animals in their natural habitat during golden hour",
    "a diverse group of students from different countries share their cultural experiences in the university campus",
    "the old bookstore with wooden shelves filled with rare books and antique manuscripts attracts curious visitors",
    "professional surfers ride massive waves while spectators watch in amazement from the sandy beach",
    "experienced hikers trek through dense forest following ancient trails marked by indigenous communities",
    "the bustling night market filled with street food vendors cooking traditional dishes while locals and tourists browse colorful stalls under lantern light",
    "talented street artists create impressive murals on city walls using vibrant colors and innovative techniques to tell stories about local culture",
    "dedicated scientists working in advanced laboratories conduct groundbreaking research using state-of-the-art equipment to solve complex medical problems",
    "the annual cultural festival brings together people from diverse backgrounds to celebrate with traditional music performances dance shows art exhibitions and food stalls representing different regions",
    "skilled craftsmen in the traditional workshop carefully restore antique furniture using time-honored techniques passed down through generations while teaching apprentices their valuable skills",
    "the innovative sustainable urban farm project combines modern hydroponics technology with traditional farming methods to grow organic vegetables fruits and herbs while educating local communities about sustainable agriculture practices",
    "professional wildlife photographers spend months in remote locations documenting rare animal species and their behaviors while working with conservation teams to protect endangered habitats and raise awareness"
]


class BaseBenchmark:
    def __init__(self):
        self.all_images = copy.deepcopy(_DEFAULT_IMAGE_POOL)
        self.all_texts = copy.deepcopy(_DEFAULT_TEXT_POOL)

    def prepare(self):
        pass

    def load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def run_benchmark(self, run_times):
        logs = []
        current_process = psutil.Process()

        def _record(name):
            logs.append((name, current_process.memory_info().rss, time.time()))

        # make sure the model is downloaded
        self.prepare()
        self.load()
        self.unload()

        _record('<init>')

        self.load()
        _record('<load>')

        for i in tqdm(range(run_times)):
            self.run()
            _record(f'#{i + 1}')

        self.unload()
        _record('<unload>')

        mems = np.array([mem for _, mem, _ in logs])
        mems -= mems[0]
        times = np.array([time_ for _, _, time_ in logs])
        times -= times[0]
        times[1:] = times[1:] - times[:-1]
        labels = np.array([name for name, _, _ in logs])

        return mems, times, labels

    def _run_in_subprocess_share(self, run_times, ret):
        ret['retval'] = self.run_benchmark(run_times)

    def run_in_subprocess(self, run_times: int = 10, try_times: int = 10):
        manager = multiprocessing.Manager()
        full_deltas, full_times, final_labels = [], [], None
        for i in tqdm(range(try_times)):
            ret = manager.dict()
            p = Process(target=self._run_in_subprocess_share, args=(run_times, ret,))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise ChildProcessError(f'Exitcode {p.exitcode} in {self!r}\'s {ordinalize(i + 1)} try.')

            mems, times, labels = ret['retval']
            deltas = mems[1:] - mems[:-1]
            full_deltas.append(deltas)
            full_times.append(times)
            if final_labels is None:
                final_labels = labels

        deltas = np.stack(full_deltas).mean(axis=0)
        final_mems = np.cumsum([0, *deltas])
        final_times = np.stack(full_times).mean(axis=0)

        return final_mems, final_times, final_labels


def create_plot_cli(items: List[Tuple[str, BaseBenchmark]],
                    title: str = 'Unnamed Benchmark Plot', run_times=15, try_times=10,
                    mem_ylog: bool = False, time_ylog: bool = False,
                    figsize=(1080, 600), dpi: int = 300):
    def fmt_size(x, pos):
        _ = pos
        warnings.filterwarnings('ignore')
        return size_to_bytes_str(x, precision=1)

    def fmt_time(x, pos):
        _ = pos
        if x < 1e-6:
            return f'{x * 1e9:.1f}ns'
        elif x < 1e-3:
            return f'{x * 1e6:.1f}μs'
        elif x < 1:
            return f'{x * 1e3:.1f}ms'
        else:
            return f'{x * 1.0:.1f}s'

    @click.command()
    @click.option('--output', '-o', 'save_as', type=click.Path(dir_okay=False), required=True,
                  help='Output path of image file.', show_default=True)
    def _execute(save_as):
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] / INCHES_TO_PIXELS, figsize[1] / INCHES_TO_PIXELS))

        if mem_ylog:
            axes[0].set_yscale('log')
        axes[0].yaxis.set_major_formatter(FuncFormatter(fmt_size))
        axes[0].set_title('Memory Benchmark')
        axes[0].set_ylabel('Memory Usage')

        if time_ylog:
            axes[1].set_yscale('log')
        axes[1].yaxis.set_major_formatter(FuncFormatter(fmt_time))
        axes[1].set_title('Performance Benchmark (CPU)')
        axes[1].set_ylabel('Time Cost')

        labeled = False

        for name, bm in tqdm(items):
            mems, times, labels = bm.run_in_subprocess(run_times, try_times)
            axes[0].plot(mems, label=name)
            axes[1].plot(times, label=name)
            if not labeled:
                axes[0].set_xticks(range(len(labels)), labels, rotation='vertical')
                axes[1].set_xticks(range(len(labels)), labels, rotation='vertical')
                labeled = True

        axes[0].legend()
        axes[0].grid()
        axes[1].legend()
        axes[1].grid()

        fig.suptitle(f'{title}\n'
                     f'(Mean of {plural_word(try_times, "try")}, '
                     f'run for {plural_word(run_times, "time")})')

        fig.tight_layout()
        plt.savefig(save_as, bbox_inches='tight', dpi=dpi, transparent=True)

    return _execute
