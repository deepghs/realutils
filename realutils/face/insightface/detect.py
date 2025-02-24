"""
Face detection module based on RetinaFace.

This module provides functionality for detecting faces in images using the RetinaFace model.
It includes utilities for processing images, detecting faces, and handling keypoints.
The implementation supports different model sizes and configurations through Hugging Face's model hub.
"""

# -*- coding: utf-8 -*-
from __future__ import division

import json
from typing import Tuple, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from imgutils.data import ImageTyping, load_image
from imgutils.utils import open_onnx_model, ts_lru_cache

from .base import _REPO_ID, _DEFAULT_MODEL, Face


@ts_lru_cache()
def _open_ref_info(model_name: str):
    """
    Load reference information for the specified model from JSON file.

    :param model_name: Name of the model to load reference information for
    :type model_name: str
    :return: Dictionary containing model reference information
    :rtype: dict
    """
    with open(hf_hub_download(
            repo_id=_REPO_ID,
            repo_type='model',
            filename=f'{model_name}/ref.json'
    ), 'r') as f:
        return json.load(f)


@ts_lru_cache()
def _open_det_model(model_name: str):
    """
    Load and initialize the detection model.

    :param model_name: Name of the model to load
    :type model_name: str
    :return: Tuple of (model session, input name, output names)
    :rtype: tuple
    """
    model_filename = _open_ref_info(model_name)['det']
    session = open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename=f'{model_name}/{model_filename}'
    ))
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names


def distance2bbox(points, distance):
    """
    Convert distance predictions to bounding box coordinates.

    :param points: Center points of anchors
    :type points: numpy.ndarray
    :param distance: Distance predictions from model
    :type distance: numpy.ndarray
    :return: Bounding box coordinates (x1, y1, x2, y2)
    :rtype: numpy.ndarray
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """
    Convert distance predictions to keypoint coordinates.

    :param points: Center points of anchors
    :type points: numpy.ndarray
    :param distance: Distance predictions for keypoints
    :type distance: numpy.ndarray
    :return: Keypoint coordinates
    :rtype: numpy.ndarray
    """
    preds = []
    for idx in range(0, distance.shape[1], 2):
        px = points[:, idx % 2] + distance[:, idx]
        py = points[:, (idx % 2) + 1] + distance[:, idx + 1]
        preds.extend([px, py])
    return np.stack(preds, axis=-1)


def nms(dets, nms_thresh: float = 0.4):
    """
    Perform non-maximum suppression on detection results.

    :param dets: Detection results including scores
    :type dets: numpy.ndarray
    :param nms_thresh: NMS threshold
    :type nms_thresh: float
    :return: Indices of kept detections
    :rtype: list
    """
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(ovr <= nms_thresh)[0] + 1]

    return keep


_FEAT_STRIDE_FPN = [8, 16, 32]
_FMC = len(_FEAT_STRIDE_FPN)
_NUM_ANCHORS = 2


@ts_lru_cache()
def _get_center_from_cache(key):
    """
    Get anchor centers from cache or compute if not cached.

    :param key: Tuple of (height, width, stride)
    :type key: tuple
    :return: Computed anchor centers
    :rtype: numpy.ndarray
    """
    height, width, stride = key
    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32) * stride
    anchor_centers = anchor_centers.reshape(-1, 2)
    if _NUM_ANCHORS > 1:
        anchor_centers = np.stack([anchor_centers] * _NUM_ANCHORS, axis=1).reshape((-1, 2))
    return anchor_centers


def _det_inference(img_data: np.ndarray, threshold: float, model_name: str = _DEFAULT_MODEL):
    """
    Perform face detection inference on image data.

    :param img_data: Input image data
    :type img_data: numpy.ndarray
    :param threshold: Detection confidence threshold
    :type threshold: float
    :param model_name: Name of the model to use
    :type model_name: str
    :return: Tuple of (scores_list, bboxes_list, kpss_list)
    :rtype: tuple
    """
    session, input_name, output_names = _open_det_model(model_name=model_name)
    blob = (np.array(img_data, dtype=np.float32) - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    net_outs = session.run(output_names, {input_name: blob})
    scores_list, bboxes_list, kpss_list = [], [], []

    for idx, stride in enumerate(_FEAT_STRIDE_FPN):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + _FMC] * stride
        kps_preds = net_outs[idx + _FMC * 2] * stride

        height, width = blob.shape[2] // stride, blob.shape[3] // stride
        key = (height, width, stride)
        center = _get_center_from_cache(key)

        pos_inds = np.where(scores >= threshold)[0]
        bboxes = distance2bbox(center, bbox_preds)
        scores_list.append(scores[pos_inds])
        bboxes_list.append(bboxes[pos_inds])
        kpss = distance2kps(center, kps_preds).reshape(len(bboxes), -1, 2)
        kpss_list.append(kpss[pos_inds])

    return scores_list, bboxes_list, kpss_list


def isf_detect_faces(image: ImageTyping, model_name: str = _DEFAULT_MODEL,
                     input_size: Tuple[int, int] = (640, 640),
                     det_thresh: float = 0.5, nms_thresh: float = 0.4) -> List[Face]:
    """
    Detect faces in the given image using RetinaFace model.

    :param image: Input image (can be path, PIL Image, or numpy array)
    :type image: Union[str, PIL.Image.Image, numpy.ndarray]
    :param model_name: Name of the detection model to use
    :type model_name: str
    :param input_size: Model input size (width, height)
    :type input_size: tuple
    :param det_thresh: Detection confidence threshold
    :type det_thresh: float
    :param nms_thresh: Non-maximum suppression threshold
    :type nms_thresh: float
    :return: List of detected faces with bounding boxes and keypoints
    :rtype: List[Face]

    :example:
        >>> from realutils.face.insightface import isf_detect_faces
        >>> from PIL import Image
        >>>
        >>> img = Image.open('path/to/image.jpg')
        >>> faces = isf_detect_faces(img)
        >>> for face in faces:
        ...     print(f"Face {face.bbox!r} detected with confidence: {face.det_score}")
    """
    pil_img = load_image(image, force_background='white', mode='RGB')
    im_ratio = pil_img.height / pil_img.width
    model_ratio = input_size[1] / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)

    det_scale = new_height / pil_img.height
    resized = pil_img.resize((new_width, new_height), Image.BILINEAR)
    det_img = Image.new('RGB', input_size, (0, 0, 0))
    det_img.paste(resized, (0, 0))

    scores_list, bboxes_list, kpss_list = _det_inference(
        img_data=np.array(det_img),
        model_name=model_name,
        threshold=det_thresh,
    )
    scores = np.vstack(scores_list)
    order = scores.ravel().argsort()[::-1]

    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale

    pre_det = np.hstack([bboxes, scores]).astype(np.float32)[order]
    keep = nms(pre_det, nms_thresh=nms_thresh)
    det = pre_det[keep]
    kpss = kpss[order][keep] if kpss is not None else None
    assert det.shape[0] == kpss.shape[0], \
        (f'Detections and keypoints first dimension not match, '
         f'{det.shape!r} vs {kpss.shape!r}. '
         f'This should be a bug, please file an issue.')

    retval = []
    for i in range(det.shape[0]):
        bbox = det[i, 0:4]
        det_score = det[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]

        retval.append(Face(
            bbox=tuple(bbox.tolist()),
            det_score=det_score.item(),
            keypoints=[tuple(item) for item in kps.tolist()],
        ))

    return retval
