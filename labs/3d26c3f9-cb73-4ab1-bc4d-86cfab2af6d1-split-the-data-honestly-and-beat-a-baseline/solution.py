import numpy as np

def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """
    Splits dataset into train, validation, and test folds honestly using a random seed,
    fits a baseline model on the training fold ONLY, and predicts on the test fold.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_end = n_train
    val_end = train_end + n_val
    train_idx = shuffled_indices[:train_end]
    val_idx = shuffled_indices[train_end:val_end]
    test_idx = shuffled_indices[val_end:]
    y_train = y[train_idx]
    train_mean = np.mean(y_train)
    n_test = len(test_idx)
    test_predictions = np.full(n_test, train_mean)
    return test_predictions, train_idx, val_idx, test_idx
