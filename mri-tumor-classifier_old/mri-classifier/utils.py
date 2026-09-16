import numpy as np

def threshold(logits, threshold=0.5, value_min=0, value_max=1):
    x = np.array(logits)
    x[x >= threshold] = value_max
    x[x < threshold] = value_min
    return x
    
