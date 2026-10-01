import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    x = np.array(x, dtype=float)
    n = len(x)                          # number of inputs
    m = len(f(list(x)))                 # number of outputs

    J = np.zeros((m, n))
    for j in range(n):
        step = np.zeros(n)
        step[j] = h
        f_plus = np.array(f(list(x + step)), dtype=float)
        f_minus = np.array(f(list(x - step)), dtype=float)
        J[:, j] = (f_plus - f_minus) / (2 * h)   # column j = d(outputs)/d(x_j)

    return J.tolist()