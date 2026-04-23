from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def evaluate_batch(model, X_batch, y_batch):
    """
    Evaluates model on a batch and returns detailed metrics.
    """
    y_pred = model.predict(X_batch)

    n_classes = len(np.unique(y_batch))
    average = "binary" if n_classes <= 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_batch, y_pred),
        "precision": precision_score(y_batch, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_batch, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_batch, y_pred, average=average, zero_division=0),
    }

    return metrics
