import numpy as np


def apply_covariate_drift(X, drift_strength=0.3):
    """
    Applies covariate drift by shifting numerical feature distributions.
    drift_strength controls how much the features shift (fraction of mean).
    """
    X_drifted = X.copy()

    numeric_cols = X_drifted.select_dtypes(include="number").columns

    for col in numeric_cols:
        col_std = X_drifted[col].std()
        noise = np.random.normal(0, col_std * drift_strength * 0.1, size=len(X_drifted))
        mean_shift = X_drifted[col].mean() * drift_strength
        X_drifted[col] = X_drifted[col] + mean_shift + noise

    return X_drifted
