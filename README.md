# realutils

[![PyPI](https://img.shields.io/pypi/v/dghs-realutils)](https://pypi.org/project/dghs-realutils/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dghs-realutils)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/2df500fa7fddd97549d0e027680b9c8f/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/2df500fa7fddd97549d0e027680b9c8f/raw/comments.json)

[![Code Test](https://github.com/deepghs/realutils/workflows/Code%20Test/badge.svg)](https://github.com/deepghs/realutils/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/deepghs/realutils/workflows/Package%20Release/badge.svg)](https://github.com/deepghs/realutils/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/deepghs/realutils/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/deepghs/realutils)

[![Discord](https://img.shields.io/discord/1157587327879745558?style=social&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FTwdHJ42N72)](https://discord.gg/TwdHJ42N72)
![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/realutils)](https://github.com/deepghs/realutils/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/realutils)](https://github.com/deepghs/realutils/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/realutils)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/realutils)](https://github.com/deepghs/realutils/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/realutils)](https://github.com/deepghs/realutils/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/realutils)](https://github.com/deepghs/realutils/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/realutils)](https://github.com/deepghs/realutils/blob/master/LICENSE)

A convenient and user-friendly image data processing library that integrates various advanced image processing models.

## Installation

You can simply install it with `pip` command line from the official PyPI site.

```shell
pip install dghs-realutils
```

If your operating environment includes a available GPU, you can use the following installation command to achieve higher
performance:

```shell
pip install dghs-realutils[gpu]
```

For more information about installation, you can refer
to [Installation](https://deepghs.github.io/realutils/main/tutorials/installation/index.html).

## Supported or Developing Features

`realutils` includes many generic usable features which are available on non-GPU device.
For detailed descriptions and examples, please refer to the
[official documentation](https://deepghs.github.io/realutils/main/index.html).
Here, we won't go into each of them individually.

### Real Human Photo Tagger

We have tagger for real human photos, like this

![idolsankaku_tagger](https://github.com/deepghs/realutils/blob/main/docs/source/api_doc/tagging/idolsankaku_demo_readme.plot.py.svg)

We can use `get_idolsankaku_tags` to tag them

```python
from realutils.tagging import get_idolsankaku_tags

rating, general, character = get_idolsankaku_tags('idolsankaku/1.jpg')
print(rating)
# {'safe': 0.748395562171936, 'questionable': 0.22442740201950073, 'explicit': 0.022273868322372437}
print(general)
# {'1girl': 0.7476911544799805, 'asian': 0.3681548237800598, 'skirt': 0.8094233274459839, 'solo': 0.44033104181289673, 'blouse': 0.7909733057022095, 'pantyhose': 0.8893758654594421, 'long_hair': 0.7415428161621094, 'brown_hair': 0.4968719780445099, 'sitting': 0.49351146817207336, 'high_heels': 0.41397374868392944, 'outdoors': 0.5279690623283386, 'non_nude': 0.4075928330421448}
print(character)
# {}

rating, general, character = get_idolsankaku_tags('idolsankaku/7.jpg')
print(rating)
# {'safe': 0.9750080704689026, 'questionable': 0.0257779061794281, 'explicit': 0.0018109679222106934}
print(general)
# {'1girl': 0.5759814381599426, 'asian': 0.46296364068984985, 'skirt': 0.9698911905288696, 'solo': 0.6263223886489868, 'female': 0.5258357524871826, 'blouse': 0.8670071959495544, 'twintails': 0.9444552659988403, 'pleated_skirt': 0.8233045935630798, 'miniskirt': 0.8354354500770569, 'long_hair': 0.8752110004425049, 'looking_at_viewer': 0.4927205741405487, 'detached_sleeves': 0.9382797479629517, 'shirt': 0.8463951945304871, 'tie': 0.8901710510253906, 'aqua_hair': 0.9376567006111145, 'armpit': 0.5968506336212158, 'arms_up': 0.9492673873901367, 'sleeveless_blouse': 0.9789504408836365, 'black_thighhighs': 0.41496211290359497, 'sleeveless': 0.9865490198135376, 'default_costume': 0.36392033100128174, 'sleeveless_shirt': 0.9865082502365112, 'very_long_hair': 0.3988983631134033}
print(character)
# {'hatsune_miku': 0.9460012912750244}
```

For more details,
see: [documentation of get_idolsankaku_tags](https://dghs-realutils.deepghs.org/main/api_doc/tagging/idolsankaku.html#get-idolsankaku-tags).

### Generic Object Detection

We use official YOLO models the generic purpose of object detections.

![object_detection](https://github.com/deepghs/realutils/blob/gh-pages/main/_images/yolo_demo.plot.py.svg)

We can use `detect_by_yolo` for generic object detection

```python
from realutils.detect import detect_by_yolo

print(detect_by_yolo('yolo/unsplash_aJafJ0sLo6o.jpg'))
# [((450, 317, 567, 599), 'person', 0.9004617929458618)]
print(detect_by_yolo('yolo/unsplash_n4qQGOBgI7U.jpg'))
# [((73, 101, 365, 409), 'vase', 0.9098997116088867), ((441, 215, 659, 428), 'vase', 0.622944176197052), ((5, 1, 428, 377), 'potted plant', 0.5178268551826477)]
print(detect_by_yolo('yolo/unsplash_vUNQaTtZeOo.jpg'))
# [((381, 103, 676, 448), 'bird', 0.9061452150344849)]
print(detect_by_yolo('yolo/unsplash_YZOqXWF_9pk.jpg'))
# [((315, 100, 690, 532), 'horse', 0.9453459978103638), ((198, 181, 291, 256), 'horse', 0.917123556137085), ((145, 173, 180, 249), 'horse', 0.7972317337989807), ((660, 138, 701, 170), 'horse', 0.4843617379665375)]
```

More models are hosted on [huggingface repository](https://huggingface.co/deepghs/yolos).
An online demo are provided as well, you can try [it](https://huggingface.co/spaces/deepghs/yolos) out.

### Face Detection

We use YOLO models from [akanametov/yolo-face](https://github.com/akanametov/yolo-face) for face detection.

![face_detection](https://github.com/deepghs/realutils/blob/gh-pages/main/_images/face_detect_demo.plot.py.svg)

We can use `detect_real_faces` for face detection

```python
from realutils.detect import detect_real_faces

print(detect_real_faces('yolo/solo.jpg'))
# [((168, 79, 245, 199), 'face', 0.7996422052383423)]
print(detect_real_faces('yolo/2girls.jpg'))
# [((721, 152, 1082, 726), 'face', 0.8811314702033997), ((158, 263, 509, 714), 'face', 0.8745490908622742)]
print(detect_real_faces('yolo/3+cosplay.jpg'))
# [((351, 228, 410, 302), 'face', 0.8392542600631714), ((384, 63, 427, 116), 'face', 0.8173024654388428), ((195, 109, 246, 161), 'face', 0.8126493692398071)]
print(detect_real_faces('yolo/multiple.jpg'))
# [((1074, 732, 1258, 987), 'face', 0.8792377710342407), ((1378, 536, 1541, 716), 'face', 0.8607611656188965), ((554, 295, 759, 557), 'face', 0.8541485071182251), ((897, 315, 1068, 520), 'face', 0.8539882898330688), ((1194, 230, 1329, 403), 'face', 0.8324605226516724)]
```

More models are hosted on [huggingface repository](https://huggingface.co/deepghs/yolo-face).
An online demo are provided as well, you can try [it](https://huggingface.co/spaces/deepghs/yolo-face) out.

### Feature Extractor

We support DINOv2-based image feature extractor, like this

```python
from realutils.metrics import get_dinov2_embedding

embedding = get_dinov2_embedding('unsplash_0aLd44ICcpg.jpg')
print(embedding.shape)
# (768,)
```

You can use this embedding, calculating their cosine similarities to measure their visual similarities.

### Image-Text Models

We support both CLIP and SigLIP for multimodal alignment operations, like this

* CLIP

```python
from realutils.metrics.clip import classify_with_clip

print(classify_with_clip(
    images=[
        'xlip/1.jpg',
        'xlip/2.jpg'
    ],
    texts=[
        'a photo of a cat',
        'a photo of a dog',
        'a photo of a human',
    ],
))
# array([[0.98039913, 0.00506729, 0.01453355],
#       [0.05586662, 0.02006196, 0.92407143]], dtype=float32)
```

* SigLIP

```python
from realutils.metrics.siglip import classify_with_siglip

print(classify_with_siglip(
    images=[
        'xlip/1.jpg',
        'xlip/2.jpg',
    ],
    texts=[
        'a photo of a cat',
        'a photo of 2 cats',
        'a photo of 2 dogs',
        'a photo of a woman',
    ],
))
# array([[1.3782851e-03, 2.7010253e-01, 9.7517688e-05, 3.6702781e-09],
#        [3.3248414e-06, 2.2294161e-07, 1.9753381e-09, 2.2561464e-06]],
#       dtype=float32)
```

For more details, you can take a look at:

* [Documentation of realutils.metrics.clip](https://dghs-realutils.deepghs.org/main/api_doc/metrics/clip.html)
* [Models of CLIP](https://huggingface.co/deepghs/clip_onnx)
* [Documentation of realutils.metrics.siglip](https://dghs-realutils.deepghs.org/main/api_doc/metrics/siglip.html)
* [Models of SigLIP](https://huggingface.co/deepghs/siglip_onnx)

