import numpy as np


def stream_batches(X, y, n_batches=10):
    """
    Generator that yields data divided into a user-specified number of batches.
    Instead of fixed batch_size, the user controls how many batches to split into.
    """
    n_samples = len(X)
    batch_size = max(1, n_samples // n_batches)

    for i in range(n_batches):
        start = i * batch_size
        # Last batch gets all remaining samples
        if i == n_batches - 1:
            end = n_samples
        else:
            end = start + batch_size

        if start >= n_samples:
            break

        yield X.iloc[start:end], y.iloc[start:end]
