from sklearn.ensemble import RandomForestClassifier
from config import N_ESTIMATORS, RANDOM_STATE

def retrain_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model
