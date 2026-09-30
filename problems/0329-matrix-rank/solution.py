import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    M = np.array(A, dtype=float)          # work on a copy, never modify the input
    if M.ndim != 2 or M.size == 0:
        return 0

    m, n = M.shape
    rank = 0
    row = 0

    for col in range(n):
        if row >= m:
            break

        # partial pivoting: largest entry in this column at or below `row`
        pivot = row + np.argmax(np.abs(M[row:, col]))
        if abs(M[pivot, col]) <= tol:
            continue                      # no usable pivot: column depends on earlier ones

        # move the pivot row into place
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]

        # eliminate everything below the pivot
        factors = M[row + 1:, col] / M[row, col]
        M[row + 1:, :] -= np.outer(factors, M[row, :])

        row += 1
        rank += 1

    return rank
    