import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)

    n = X.shape[0]
    Xc = X - X.mean(axis=0)                 # center each column
    Yc = Y - Y.mean(axis=0)

    cov = Xc.T @ Yc / n                     # (features_X, features_Y)
    std_x = X.std(axis=0)                   # population std (divides by n)
    std_y = Y.std(axis=0)

    return cov / np.outer(std_x, std_y)  