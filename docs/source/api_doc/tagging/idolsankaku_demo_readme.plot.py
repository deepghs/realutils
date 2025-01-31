import io
import os.path

from tqdm import tqdm

from plot import image_plot
from realutils.tagging import get_idolsankaku_tags


def _make_label(file):
    with io.StringIO() as sf:
        print(file, file=sf)
        rating, general, character = get_idolsankaku_tags(file)
        rt, srt = None, None
        for r, sr in rating.items():
            if rt is None or sr > srt:
                rt, srt = r, sr
        print(f'{rt}: {srt:.4f}', file=sf)
        seg_cnt = 6
        print('Tags:', end=' ', file=sf)
        for i, tag in enumerate(general.keys()):
            print(f'{tag}{", " if i < len(general) - 1 else ""}', end='', file=sf)
            seg_cnt += len(tag) + (2 if i < len(general) - 1 else 0)
            if seg_cnt >= 28:
                print('', file=sf)
                seg_cnt = 0
        if seg_cnt > 0:
            print('', file=sf)
        if character:
            print(f'Character: {", ".join(character.keys())}', file=sf)

        return sf.getvalue()


if __name__ == '__main__':
    image_plot(
        *[
            (file, _make_label(file))
            for file in tqdm([
                os.path.join('idolsankaku', '1.jpg'),
                os.path.join('idolsankaku', '2.jpg'),
                os.path.join('idolsankaku', '7.jpg')
            ])
        ],
        columns=3,
        figsize=(12, 6),
    )
