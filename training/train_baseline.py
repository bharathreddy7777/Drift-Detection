import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATASET_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.mean())

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from config import N_ESTIMATORS
def train_baseline_model():
    X_train, X_test, y_train, y_test = load_dataset()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_rf.pkl")

    return metrics
