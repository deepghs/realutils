"""
This module provides functionality for visualizing face detection results on images.
It includes utilities for drawing bounding boxes, keypoints, and labels on detected faces.
The visualization can be customized with different colors, font sizes, and display options.
"""
import math
from typing import List, Optional

from PIL import ImageDraw, ImageFont
from hbutils.color import rnd_colors, Color
from imgutils.data import ImageTyping, load_image
from imgutils.detect.visual import _try_get_font_from_matplotlib

from .base import Face


def isf_faces_visualize(image: ImageTyping, faces: List[Face], text_padding: int = 6, fontsize: int = 12,
                        keypoint_size: int = 12, box_color: str = '#ff00ee', max_short_edge_size: Optional[int] = None,
                        fp=None, no_label: bool = False):
    """
    Visualize face detection results by drawing bounding boxes, keypoints and labels on an image.

    This function takes an input image and a list of detected faces, then draws:

    - Bounding boxes around each detected face
    - Keypoints for facial features using randomly generated colors
    - Optional labels showing detection confidence scores

    The visualization can be customized through various parameters like font size,
    box colors, and whether to show labels.

    :param image: Input image to visualize detections on. Can be a PIL Image, numpy array, or path to image file.
    :type image: ImageTyping
    :param faces: List of detected Face objects containing bounding boxes, keypoints and scores.
    :type faces: List[Face]
    :param text_padding: Padding around label text in pixels.
    :type text_padding: int
    :param fontsize: Font size for label text.
    :type fontsize: int
    :param keypoint_size: Size of keypoint markers in pixels.
    :type keypoint_size: int
    :param box_color: Color of bounding box in hex format (e.g. '#ff00ee').
    :type box_color: str
    :param max_short_edge_size: Maximum size of shortest image edge. If specified, image will be resized
        while maintaining aspect ratio.
    :type max_short_edge_size: Optional[int]
    :param fp: Font properties for matplotlib font. Only used if matplotlib is available.
    :type fp: matplotlib.font_manager.FontProperties or None
    :param no_label: If True, suppresses drawing of labels.
    :type no_label: bool

    :return: PIL Image with visualized detection results.
    :rtype: PIL.Image.Image

    :example:
        >>> from realutils.face.insightface import isf_faces_visualize, isf_detect_faces
        >>> from PIL import Image
        >>>
        >>> image = Image.open('face.jpg')
        >>> faces = isf_detect_faces(image)
        >>> visualized = isf_faces_visualize(
        ...     image,
        ...     faces,
        ...     fontsize=14,
        ...     box_color='#00ff00'
        ... )
        >>> visualized.show()
    """
    image = load_image(image, force_background=None, mode='RGBA')
    original_width, original_height = image.width, image.height
    if max_short_edge_size is not None and max_short_edge_size < min(original_height, original_width):
        r = max_short_edge_size / min(original_height, original_width)
        new_width = int(math.ceil(original_width * r))
        new_height = int(math.ceil(original_height * r))
    else:
        new_width, new_height = original_width, original_height

    visual_image = image.copy()
    if (new_width, new_height) != (original_width, original_height):
        visual_image = visual_image.resize((new_width, new_height))
    draw = ImageDraw.Draw(visual_image, mode='RGBA')
    font = _try_get_font_from_matplotlib(fp, fontsize) or ImageFont.load_default()

    kps_count = max([len(face.keypoints) for face in faces]) if faces else 5
    kps_colors = list(map(str, rnd_colors(kps_count)))
    for _, face in enumerate(faces):
        (x0, y0, x1, y1), label, score = face.to_det_tuple()
        x0, y0 = int(x0 * new_width / original_width), int(y0 * new_height / original_height)
        x1, y1 = int(x1 * new_width / original_width), int(y1 * new_height / original_height)
        keypoints = face.keypoints
        keypoints = [
            (int(x * new_width / original_width), int(y * new_height / original_height))
            for x, y in keypoints
        ]

        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=2)
        for ki, (kx, ky) in enumerate(keypoints):
            draw.ellipse(
                (
                    kx - keypoint_size / 2, ky - keypoint_size / 2,
                    kx + keypoint_size / 2, ky + keypoint_size / 2,
                ),
                fill=kps_colors[ki],
                outline=kps_colors[ki],
            )

        if not no_label:
            label_text = f'{label}: {score * 100:.2f}%'
            _t_x0, _t_y0, _t_x1, _t_y1 = draw.textbbox((x0, y0), label_text, font=font)
            _t_width, _t_height = _t_x1 - _t_x0, _t_y1 - _t_y0
            if y0 - _t_height - text_padding < 0:
                _t_text_rect = (x0, y0, x0 + _t_width + text_padding * 2, y0 + _t_height + text_padding * 2)
                _t_text_co = (x0 + text_padding, y0 + text_padding)
            else:
                _t_text_rect = (x0, y0 - _t_height - text_padding * 2, x0 + _t_width + text_padding * 2, y0)
                _t_text_co = (x0 + text_padding, y0 - _t_height - text_padding)

            draw.rectangle(_t_text_rect, fill=str(Color(box_color, alpha=0.5)))
            draw.text(_t_text_co, label_text, fill="black", font=font)

    return visual_image
