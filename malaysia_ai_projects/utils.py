import logging


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
