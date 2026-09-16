import numpy as np


def threshold(probs, thresh=0.5, value_min=0, value_max=1):
    """Binarize an array of probabilities around a cutoff."""
    x = np.array(probs)
    out = np.where(x >= thresh, value_max, value_min)
    return out
