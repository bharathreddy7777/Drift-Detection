from sklearn.ensemble import RandomForestClassifier
from config_copy import N_ESTIMATORS, RANDOM_STATE, MAX_DEPTH, MAX_SAMPLES


def retrain_model(X_train, y_train):
    """
    Retrains a RandomForest model on the given data (sliding window).
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model
