import torch
import re
from . import utils
from . import commons
from .model_infer import SynthesizerTrn
from herpetologist import check_type
from typing import Callable

_pad = ''
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_numbers = '0123456789'
_small_letters = 'abcdefghijklmnopqrstuvwxyz'
_rejected = '\'():;"'
_punct = ':;,.?'

TTS_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)


def put_spacing_num(string):
    """
    'ni1996' -> 'ni 1996'
    """
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string)
    return re.sub(r'[ ]+', ' ', string).strip()


def text_to_sequence(text):
    r = [TTS_SYMBOLS.index(c) for c in text if c in TTS_SYMBOLS]
    return r


def text_normalization(string, norm_function=None):
    string = re.sub(r'[ ]+', ' ', string).strip()
    if string[-1] in '-,':
        string = string[:-1]
    if string[-1] not in '.,?!':
        string = string + '.'

    if norm_function:
        string = norm_function(string)

    string = put_spacing_num(string)
    string = ''.join([c for c in string if c in TTS_SYMBOLS])
    string = re.sub(r'[ ]+', ' ', string).strip()
    return string


def get_text(text, hps, norm_function=None):
    text = text_normalization(text, norm_function=norm_function)
    text_norm = text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    decoded = [TTS_SYMBOLS[t] for t in text_norm]
    text_norm = torch.LongTensor(text_norm)
    return text_norm, text, decoded


_available_models = {
    'yasmin': {
        'Size (MB)': 139,
    },
    'osman': {
        'Size (MB)': 139,
    },
}

repo_ids = {
    'yasmin': 'malay-huggingface/VITS-Yasmin',
    'osman': 'malay-huggingface/VITS-Osman',
}
checkpoint_filenames = {
    'yasmin': 'yasmin.pth',
    'osman': 'osman.pth',
}
config_filenames = {
    'yasmin': 'yasmin.json',
    'osman': 'osman.json',
}


def available_model():
    """
    List available Malay VITS models.
    """
    from malaysia_ai_projects.utils import describe_availability
    return describe_availability(_available_models)


@ check_type
def load(model: str = 'osman'):
    """
    Load Malay VITS model.

    Parameters
    ----------
    model : str, optional (default='osman')
        Model architecture supported. Allowed values:

        * ``'osman'`` - VITS Osman speaker.
        * ``'yasmin'`` - VITS Yasmin speaker.

    Returns
    -------
    result : malaysia_ai_projects.malay_vits.Model class
    """

    model = model.lower()
    if model not in _available_models:
        raise ValueError(
            'model not supported, please check supported models from `malaysia_ai_projects.malay_vits.available_model()`.'
        )

    from huggingface_hub import hf_hub_download

    config = hf_hub_download(repo_id=repo_ids[model], filename=config_filenames[model])
    model = hf_hub_download(repo_id=repo_ids[model], filename=checkpoint_filenames[model])
    return Model(model=model, config=config)


class Model:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.initialize()

    def initialize(self):
        hps = utils.get_hparams_from_file(self.config)
        self.hps = hps
        self.net_g = SynthesizerTrn(
            len(TTS_SYMBOLS),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        _ = self.net_g.eval()
        self.net_g.load_state_dict(torch.load(self.model))

    def predict(
        self,
        input: str,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        norm_function: Callable = None
    ):
        """
        Parameters
        ----------
        input: str
        noise_scale: float, optional (default=0.667)
        noise_scale_w: float, optional (default=0.8)
        length_scale: float, optional (default=1.0)
        norm_function: Callable, optional (default=None)

        Returns
        -------
        result: (audio with 22050 sample rate, text, list of chars, alignment)
        """
        stn_tst, text, decoded = get_text(input, self.hps, norm_function=norm_function)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = self.net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        alignment = audio[1].detach().numpy()[0, 0]
        audio = audio[0].detach().numpy()[0, 0]
        return audio, text, decoded, alignment
