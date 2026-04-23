import numpy as np


def apply_concept_drift(y, drift_ratio=0.3):
    """
    Simulates concept drift by flipping a fraction of labels.
    drift_ratio controls what percentage of labels get flipped.
    """
    y_drifted = y.copy()
    n_flip = int(len(y) * drift_ratio)

    flip_indices = np.random.choice(
        y.index, size=n_flip, replace=False
    )

    y_drifted.loc[flip_indices] = 1 - y_drifted.loc[flip_indices]

    return y_drifted
