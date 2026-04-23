import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from config_copy import N_ESTIMATORS, RANDOM_STATE, MAX_DEPTH, MAX_SAMPLES, MAX_TRAIN_ROWS


def load_and_prepare_dataset(df, target_column, test_size=0.3):
    """
    Accepts a DataFrame and target column name (selected at runtime by the user).
    Encodes categoricals, handles missing values, and returns train-test split.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target if categorical
    if y.dtype == "object":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
        label_encoders["__target__"] = le_target

    # Handle missing values
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # Cap training data to avoid memory issues
    if len(X_train) > MAX_TRAIN_ROWS:
        X_train = X_train.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]

    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, X_test, y_train, y_test):
    """
    Trains a baseline Random Forest model and returns metrics + trained model.
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n_classes = len(y_test.unique())
    average = "binary" if n_classes <= 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    return model, metrics
