import numpy as np
import onnx
import torch
from PIL import Image
from ditk import logging
from onnxruntime import InferenceSession
from transformers import AutoImageProcessor, AutoModel


def onnx_export(save_path: str, model_name: str = 'facebook/dinov2-base'):
    logging.info(f'Try exporting {model_name!r} to {save_path!r} ...')
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_image = Image.new('RGB', (640, 640), color='black')
    dummy_shape = processor(images=dummy_image, return_tensors="pt")
    dummy_input = torch.randn_like(dummy_shape['pixel_values'])

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        custom_opsets=None,
    )

    def verify_onnx():
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(save_path)
        session = InferenceSession(save_path)
        input_name = session.get_inputs()[0].name
        input_data = dummy_input.numpy()

        logging.info('Inference with onnx model ...')
        onnx_outputs, px = session.run(None, {input_name: input_data})
        logging.info('Inference with torch model ...')
        with torch.no_grad():
            torch_outputs = model(dummy_input)

        np.testing.assert_allclose(torch_outputs.pooler_output.numpy(), px, rtol=1e-03, atol=1e-04)
        logging.info('Export validate success.')

        logging.info('Model input information:')
        for input in session.get_inputs():
            logging.info(f"    name: {input.name!r}, shape: {input.shape!r}, type: {input.type!r}")
        logging.info('Model output information:')
        for output in session.get_outputs():
            logging.info(f"    name: {output.name!r}, shape: {output.shape!r}, type: {output.type!r}")

        return px.shape[-1]

    return verify_onnx()
