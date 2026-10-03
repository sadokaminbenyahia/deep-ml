import numpy as np

def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("invalid input types")
    if y_true.shape != y_pred.shape:
        raise ValueError("not the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    return round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3)