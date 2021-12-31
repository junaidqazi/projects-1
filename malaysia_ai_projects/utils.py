import logging
import numpy as np


def describe_availability(dict, transpose=True, text=''):
    if len(text):
        logging.basicConfig(level=logging.INFO)

        logging.info(text)
    try:
        import pandas as pd

        df = pd.DataFrame(dict)

        if transpose:
            return df.T
        else:
            return df
    except BaseException:
        logging.info('pandas not installed, will returned dictionary instead.')
        return dict


def imagenet_input(x):
    x = x.copy()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
    return x
