from ditk import logging

from .model import sync

if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repository='deepghs/timms_mobilenet',
        params_limit=0.03 * 1000 ** 3,
        max_count=100,
        name_filter=lambda x: 'mobilenet' in x,
    )
