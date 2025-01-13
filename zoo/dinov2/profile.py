import logging

import torch
from PIL import Image
from thop import profile, clever_format
from transformers import AutoImageProcessor, AutoModel


def model_profile(model_name: str = 'facebook/dinov2-base'):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_image = Image.new('RGB', (640, 640), color='black')
    dummy_shape = processor(images=dummy_image, return_tensors="pt")
    dummy_input = torch.randn_like(dummy_shape['pixel_values'])

    macs, params = profile(model, inputs=(dummy_input,))
    s_macs, s_params = clever_format([macs, params], "%.1f")
    logging.info(f'Model profile, params: {s_params}, flops: {s_macs}')
    return {
        'flops': macs,
        'params': params,
    }
