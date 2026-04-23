def apply_covariate_drift(X, drift_strength=0.3):
    """
    Applies covariate drift by shifting numerical feature distributions
    """
    X_drifted = X.copy()

    numeric_cols = X_drifted.select_dtypes(include="number").columns

    for col in numeric_cols:
        mean_shift = X_drifted[col].mean() * drift_strength
        X_drifted[col] = X_drifted[col] + mean_shift

    return X_drifted
