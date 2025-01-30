from ditk import logging

from .export import export

if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    export('yolov8n-person', repository='deepghs/yolo-person')
