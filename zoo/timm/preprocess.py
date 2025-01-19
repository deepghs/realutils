from timm.data import MaybeToTensor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize


def _iter_trans(x):
    if isinstance(x, Compose):
        for xitem in x.transforms:
            yield from _iter_trans(xitem)
    elif isinstance(x, Resize):
        yield {
            'type': 'resize',
            'size': x.size,
            'max_size': x.max_size,
            'interpolation': x.interpolation.value,
            'antialias': x.antialias,
        }
    elif isinstance(x, CenterCrop):
        yield {
            'type': 'center_crop',
            'size': x.size,
        }
    elif isinstance(x, Normalize):
        yield {
            'type': 'normalize',
            'mean': x.mean.tolist(),
            'std': x.std.tolist(),
        }
    elif isinstance(x, MaybeToTensor):
        yield {
            'type': 'maybe_to_tensor',
        }
    else:
        raise RuntimeError(f'Unknown Transform - {x!r}')


def export_trans(x):
    return list(_iter_trans(x))
