import logging
import numpy as np
from herpetologist import check_type
from malaya_boilerplate.frozen_graph import (
    generate_session,
    nodes_session,
    load_graph,
)
from malaya_boilerplate.execute import execute_graph
from skimage.transform import resize
from typing import List

_available_models = {
    'efficientnet-b4': {
        'Size (MB)': 79.9,
        'Test Loss': 0.11,
    },
    'efficientnet-b4-quantized': {
        'Size (MB)': 20.7,
        'Test Loss': 0.11,
    },
    'efficientnet-b2': {
        'Size (MB)': 66.4,
        'Test Loss': 0.15,
    },
    'efficientnet-b2-quantized': {
        'Size (MB)': 17.1,
        'Test Loss': 0.15,
    },
}

repo_id = 'malay-huggingface/pembalakan'
huggingface_filenames = {
    'efficientnet-b4': 'efficientnet-b4/frozen_model.pb',
    'efficientnet-b4-quantized': 'efficientnet-b4/frozen_model.pb.quantized',
    'efficientnet-b2': 'efficientnet-b2/frozen_model.pb',
    'efficientnet-b2-quantized': 'efficientnet-b2/frozen_model.pb.quantized'
}


def available_model():
    """
    List available Pembalakan models.
    """
    from malaysia_ai_projects.utils import describe_availability
    return describe_availability(_available_models)


@check_type
def load(model: str = 'efficientnet-b2', **kwargs):
    """
    Load suarakami model.

    Parameters
    ----------
    model : str, optional (default='efficientnet-b2')
        Model architecture supported. Allowed values:

        * ``'efficientnet-b4'`` - EfficientNet B4 + Unet.
        * ``'efficientnet-b4-quantized'`` - EfficientNet B4 + Unet with dynamic quantized.
        * ``'efficientnet-b2'`` - EfficientNet B2 + Unet.
        * ``'efficientnet-b2-quantized'`` - EfficientNet B2 + Unet with dynamic quantized.

    Returns
    -------
    result : malaysia_ai_projects.pembalakan.Model class
    """
    model = model.lower()
    if model not in _available_models:
        raise ValueError(
            'model not supported, please check supported models from `malaysia_ai_projects.pembalakan.available_model()`.'
        )

    from huggingface_hub import hf_hub_download
    model = hf_hub_download(repo_id=repo_id, filename=huggingface_filenames[model])
    g = load_graph(package=None, frozen_graph_filename=model, **kwargs)
    input_nodes, output_nodes = nodes_session(g, ['input'], ['logits'])
    return Model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs)
    )


class Model:
    def __init__(self, input_nodes, output_nodes, sess):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._size = 256

    def _execute(self, inputs, input_labels, output_labels):
        return execute_graph(
            inputs=inputs,
            input_labels=input_labels,
            output_labels=output_labels,
            sess=self._sess,
            input_nodes=self._input_nodes,
            output_nodes=self._output_nodes,
        )

    def predict(self, inputs: List[np.array]):
        """
        Parameters
        ----------
        input: List[np.array]
            List of np.array, should be size [H, W, 3], `H` and `W` can be dynamic.

        Returns
        -------
        result: List[np.array]
        """
        sizes, batch = [], []
        for input in inputs:
            sizes.append(input.shape[:-1])
            batch.append(resize(input, (self._size, self._size), anti_aliasing=False))
        r = self._execute(
            inputs=[batch],
            input_labels=['input'],
            output_labels=['logits'],
        )
        v = r['logits']
        outputs = []
        for no, output in enumerate(v):
            outputs.append(np.around(resize(output, sizes[no], anti_aliasing=False)))
        return outputs
