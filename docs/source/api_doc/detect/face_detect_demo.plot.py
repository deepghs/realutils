from imgutils.data import load_image
from imgutils.detect.visual import detection_visualize

from plot import image_plot
from realutils.detect import detect_faces


def _detect(img, **kwargs):
    img = load_image(img, mode='RGB')
    if min(img.width, img.height) > 640:
        r = min(img.width, img.height) / 640
        new_width = int(round(img.width / r))
        new_height = int(round(img.height / r))
        img = img.resize((new_width, new_height))
    return detection_visualize(img, detect_faces(img, **kwargs))


if __name__ == '__main__':
    image_plot(
        (_detect('face/solo.jpg'), 'solo'),
        (_detect('face/2girls.jpg'), '2girls'),
        (_detect('face/3+cosplay.jpg'), '3girls cosplay'),
        (_detect('face/multiple.jpg'), 'multiple'),
        columns=2,
        figsize=(12, 9),
    )
