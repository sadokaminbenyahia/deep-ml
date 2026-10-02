import numpy as np

def to_categorical(x, n_col=None):
    x = np.asarray(x)
    if n_col is None:
        n_col = x.max() + 1             # infer number of classes
    a = np.zeros((len(x), n_col))       # shape as ONE tuple
    for j, i in enumerate(x):           # j = row number, i = class label
        a[j, i] = 1
    return a
	