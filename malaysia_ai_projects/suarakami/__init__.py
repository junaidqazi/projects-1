import logging
import string
import numpy as np
from herpetologist import check_type

labels = list(
    string.ascii_lowercase  # + string.digits
) + [' ', '_']

blank = labels.index('_')

_available_models = {
    'small-conformer': {
        'Size (MB)': 60.3,
        'WER': 0.239,
        'WER-LM': 0.14,
        'CER': 0.11,
        'CER-LM': 0.03,
        'Entropy': 0.6,
        'Language': ['malay'],
    },
    'tiny-conformer': {
        'Size (MB)': 17.9,
        'WER': 0.4,
        'WER-LM': None,
        'CER': 0.11,
        'CER-LM': None,
        'Entropy': 0.5,
        'Language': ['malay'],
    },
}

_available_lm = {
    'v1-lm': {
        'Size (MB)': 846
    },
}

repo_id = 'malay-huggingface/suarakami-models'
huggingface_filenames = {'small-conformer': 'conformer_small.onnx',
                         'tiny-conformer': 'conformer_tiny.onnx',
                         'v1-lm': 'out.trie.klm'}


def available_model():
    from malaysia_ai_projects.utils import describe_availability
    return describe_availability(_available_models)


def available_lm():
    from malaysia_ai_projects.utils import describe_availability
    return describe_availability(_available_lm)


@check_type
def load(model: str = 'small-conformer', lm: str = None):
    """
    Load suarakami model.

    Parameters
    ----------
    model : str, optional (default='small-conformer')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - Small Conformer model.

    lm: str, optional (default=None)
        Language Model supported. Allowed values:

        * ``None`` - No Language Model will use.
        * ``'v1-lm'`` - Will use V1 Language Model, size ~800 MB.

    Returns
    -------
    result : malaysia_ai_projects.suarakami.Model class
    """
    model = model.lower()
    if model not in _available_models:
        raise ValueError(
            'model not supported, please check supported models from `malaysia_ai_projects.suarakami.available_model()`.'
        )

    if isinstance(lm, str):
        lm = lm.lower()
        if lm not in _available_lm:
            raise ValueError(
                'model not supported, please check supported models from `malaysia_ai_projects.suarakami.available_lm()`.'
            )

    from huggingface_hub import hf_hub_download

    model = hf_hub_download(repo_id=repo_id, filename=huggingface_filenames[model])
    if isinstance(lm, str):
        lm = hf_hub_download(repo_id=repo_id, filename=huggingface_filenames[lm])
    return Model(model=model, lm=lm)


class Model:
    def __init__(self, model, lm):
        self.model = model
        self.lm = lm
        self.initialize()

    def initialize(self):
        import onnxruntime
        import multiprocessing

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        try:
            self.model = onnxruntime.InferenceSession(self.model, sess_options)
        except Exception as e:
            raise Exception('onnx model corrupted, please delete cache and try again.')

        if isinstance(self.lm, str):
            from pyctcdecode import build_ctcdecoder
            import kenlm

            kenlm_model = kenlm.Model(self.lm)
            self.lm = build_ctcdecoder(
                labels,
                kenlm_model,
                alpha=0.5,
                beta=1.0,
                ctc_token_idx=labels.index('_')
            )

    def decode(self, out):
        out2 = ['_']+list(out)
        collapsed = []
        for idx, i in enumerate(out):
            if i != out2[idx] and i != blank:
                collapsed.append(i)
        return ''.join([labels[i] for i in collapsed])

    def predict(self, input: np.array):
        """
        Parameters
        ----------
        input: np.array
            np.array, must in 16k rate, prefer from `librosa.load(file,16_000)`.

        Returns
        -------
        result: text, entropy, timesteps
        """
        inputs = {self.model.get_inputs()[0].name: np.expand_dims(input, 0)}
        output = self.model.run(None, inputs)[0][0]
        log_probs = output
        if self.lm is None:
            entropy = -(np.exp(log_probs) * log_probs).sum(-1).mean(-1)
            log_probs = log_probs.argmax(-1)
            text = self.decode(log_probs)
            timesteps = [0]
        else:
            out = self.lm.decode_beams(log_probs, prune_history=True)
            text, lm_state, timesteps, logit_score, lm_score = out[0]
            entropy = -(np.exp(log_probs) * log_probs).sum(-1)
            time = [i[-1] for i in timesteps]
            entropy = [entropy[i[0]:i[1]].sum().item() for i in time]
            duration = input.shape[-1] / 16_000
            mult = duration / log_probs.shape[0]
            tt = []
            for i in timesteps:
                left = i[1][0]*mult
                l = divmod(left, 1)
                left = l[0] + (l[1] * 0.06)
                right = i[1][1]*mult
                r = divmod(right, 1)
                right = r[0] + (r[1] * 0.06)
                tt.append((i[0], round(left, 2), round(right, 2)))
            timesteps = tt

        return text, entropy, timesteps
