import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    y = np.asarray(y_train, dtype=float)

    if strategy == 'mean':
        value = y.mean()
    elif strategy == 'median':
        value = np.median(y)
    elif strategy == 'quantile':
        if quantile is None or not (0 <= quantile <= 1):
            raise ValueError("quantile must be a number in [0, 1]")
        value = np.quantile(y, quantile)      # linear interpolation by default
    elif strategy == 'constant':
        if constant is None:
            raise ValueError("constant must be given when strategy='constant'")
        value = constant
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [float(value)] * n_test            # plain Python floats, not np.float64