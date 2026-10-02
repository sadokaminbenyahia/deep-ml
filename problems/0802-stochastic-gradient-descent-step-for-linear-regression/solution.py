import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n=X.shape[0]
    for i in range(n_iter):
        h=X[i%n] @ weights
        error=h-y[i%n]
        MSE=2 *error*X[i%n]
        weights=weights - learning_rate*MSE
    return weights


    pass
