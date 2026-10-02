import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = data.shape[0]
    idx = np.random.default_rng(seed).permutation(n)   # shuffled row indices, KEEP the result
    shuffled = data[idx]                                # reorder rows (fancy indexing, used on purpose here)

    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    train = shuffled[:train_end]                        # rows 0 .. train_end-1
    validation = shuffled[train_end:validation_end]     # rows train_end .. validation_end-1
    test = shuffled[validation_end:]                    # everything left
    return [train, validation, test]