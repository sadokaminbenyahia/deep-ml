import numpy as np

def dummy_classifier(y_train, n_test, strategy, constant=None):
    # Sorted unique classes and how often each appears, as plain Python lists
    classes, counts = np.unique(np.asarray(y_train), return_counts=True)
    classes = classes.tolist()   # e.g. [0, 1, 2]
    counts = counts.tolist()     # e.g. [3, 2, 1]

    if strategy == "most_frequent":
        # argmax returns the FIRST maximum; classes are sorted,
        # so ties go to the smallest label automatically
        best = classes[int(np.argmax(counts))]
        return [best] * n_test

    if strategy == "constant":
        return [constant] * n_test

    if strategy == "uniform":
        k = len(classes)
        return [classes[i % k] for i in range(n_test)]   # i % k cycles 0..k-1

    if strategy == "stratified":
        N = len(y_train)
        ideal = [n_test * c for c in counts]       # n_test * f_c, times N (kept as integers)
        alloc = [v // N for v in ideal]            # floor(n_test * f_c)
        remainder = [v % N for v in ideal]         # fractional part, times N (exact, no floats)

        leftover = n_test - sum(alloc)             # always fewer than the number of classes
        # largest remainder first; ties broken by smallest class label
        order = sorted(range(len(classes)), key=lambda j: (-remainder[j], classes[j]))
        for j in order[:leftover]:
            alloc[j] += 1

        preds = []
        for c, a in zip(classes, alloc):           # grouped, in sorted class order
            preds.extend([c] * a)
        return preds

    raise ValueError(f"Unknown strategy: {strategy}")