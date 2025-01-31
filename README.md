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

![idolsankaku_tagger]()

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



### Face Detection

We use YOLO models from []()