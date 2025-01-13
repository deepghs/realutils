import numpy as np
from PIL import Image

_DEFAULT = object()
_DEFAULT_SIZE = {"shortest_edge": 256}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def _dinov2_preprocess_image(
        image: Image.Image,
        do_resize: bool = True,
        size: dict = _DEFAULT,
        resample: int = 3,  # BICUBIC
        do_center_crop: bool = True,
        crop_size: dict = _DEFAULT,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: list = _DEFAULT,
        image_std: list = _DEFAULT,
        do_convert_rgb: bool = True
) -> np.ndarray:
    if do_convert_rgb and image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    if do_resize:
        if size is _DEFAULT:
            size = _DEFAULT_SIZE
        if 'shortest_edge' in size:
            h, w = img_array.shape[:2]
            shortest_edge = min(h, w)
            scale = size['shortest_edge'] / shortest_edge
            new_h = int(h * scale)
            new_w = int(w * scale)
        elif 'width' in size and 'height' in size:
            new_h, new_w = size['height'], size['width']
        else:
            raise ValueError(f'Invalid resize - {size!r}.')
        image = Image.fromarray(img_array)
        image = image.resize((new_w, new_h), resample=resample)
        img_array = np.array(image)

    if do_center_crop:
        if crop_size is _DEFAULT:
            crop_size = _DEFAULT_CROP_SIZE
        h, w = img_array.shape[:2]
        crop_h = crop_size['height']
        crop_w = crop_size['width']
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        img_array = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]

    if do_rescale:
        img_array = img_array * rescale_factor

    img_array = img_array.transpose(2, 0, 1)
    if do_normalize:
        if image_mean is _DEFAULT:
            image_mean = _DEFAULT_MEAN
        if image_std is _DEFAULT:
            image_std = _DEFAULT_STD
        mean = np.array(image_mean).reshape(-1, 1, 1)
        std = np.array(image_std).reshape(-1, 1, 1)
        img_array = (img_array - mean) / std

    return img_array
