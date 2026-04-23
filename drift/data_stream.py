import numpy as np

def stream_batches(X, y, batch_size=500):
    """
    Generator that yields data in sequential batches
    """
    n_samples = len(X)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        yield X.iloc[start:end], y.iloc[start:end]
