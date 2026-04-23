from training.train_baseline import train_baseline_model
from evaluation.performance_metrics import print_metrics
import joblib
from training.train_baseline import load_dataset
from drift.data_stream import stream_batches
from drift.covariate_drift import apply_covariate_drift
from evaluation.batch_evaluation import evaluate_batch
import drift.covariate_drift as cov_drift
# ---- Store results for plotting ----
batch_accuracies = []
acc_before_list = []
acc_after_list = []
ks_flags = []
psi_flags = []
error_flags = []



def run_covariate_drift_experiment():
    X_train, X_test, y_train, y_test = load_dataset()
    model = joblib.load("models/baseline_rf.pkl")

    accuracies = []

    batch_index = 0
    for X_batch, y_batch in stream_batches(X_test, y_test, batch_size=500):
        # Apply drift after 3 batches
        if batch_index >= 3:
            X_batch = apply_covariate_drift(X_batch, drift_strength=0.4)

        acc = evaluate_batch(model, X_batch, y_batch)
        accuracies.append(acc)

        print(f"Batch {batch_index} Accuracy: {acc:.4f}")
        batch_index += 1

    return accuracies
print("\nRunning Covariate Drift Experiment")
print("----------------------------------")
run_covariate_drift_experiment()
from drift.drift_detectors import ks_drift, psi_drift, error_rate_drift


def run_drift_detection():
    X_train, X_test, y_train, y_test = load_dataset()
    reference_batch = X_test.iloc[:500]
    BASELINE_ACCURACY = 0.9285
    model = joblib.load("models/baseline_rf.pkl")



    print("\nRunning Drift Detection")
    print("----------------------")

    batch_index = 0
    for X_batch, y_batch in stream_batches(X_test, y_test, batch_size=500):

        if batch_index >= 3:
            X_batch = cov_drift.apply_covariate_drift(X_batch, drift_strength=0.4)
        acc = evaluate_batch(model, X_batch, y_batch)


        ks_detected = ks_drift(reference_batch, X_batch)
        psi_detected = psi_drift(reference_batch, X_batch)
        error_drift = error_rate_drift(acc, BASELINE_ACCURACY)

        batch_accuracies.append(acc)
        ks_flags.append(int(ks_detected))
        psi_flags.append(int(psi_detected))
        error_flags.append(int(error_drift))


        print(
            f"Batch {batch_index} | "
            f"KS: {ks_detected} | "
            f"PSI: {psi_detected} | "
            f"Error Drift: {error_drift} | "
        )

        batch_index += 1
run_drift_detection()

from drift.concept_drift import apply_concept_drift
from training.retrain_model import retrain_model
import pandas as pd
from collections import deque


def run_concept_drift_and_retraining():
    X_train, X_test, y_train, y_test = load_dataset()
    model = joblib.load("models/baseline_rf.pkl")

    print("\nRunning Concept Drift + Sliding Window Retraining")
    print("------------------------------------------------")

    WINDOW_SIZE = 5
    window_X = deque(maxlen=WINDOW_SIZE)
    window_y = deque(maxlen=WINDOW_SIZE)

    batch_index = 0

    for X_batch, y_batch in stream_batches(X_test, y_test, batch_size=500):

        # Inject concept drift
        if batch_index >= 3:
            y_batch = apply_concept_drift(y_batch, drift_ratio=0.4)

        # Evaluate BEFORE any retraining
        acc_before = evaluate_batch(model, X_batch, y_batch)

        # Retrain ONLY using PREVIOUS batches
        if len(window_X) == WINDOW_SIZE:
            # print(">>> Drift confirmed. Retraining using sliding window...")
            X_retrain = pd.concat(list(window_X))
            y_retrain = pd.concat(list(window_y))
            model = retrain_model(X_retrain, y_retrain)

        # Evaluate AFTER retraining (on current unseen batch)
        acc_after = evaluate_batch(model, X_batch, y_batch)
        acc_before_list.append(acc_before)
        acc_after_list.append(acc_after)


        print(
            f"Batch {batch_index} | "
            f"Before Retrain: {acc_before:.4f} | "
            f"After Retrain: {acc_after:.4f}"
        )

        # NOW add current batch to window (for future retraining)
        window_X.append(X_batch)
        window_y.append(y_batch)

        batch_index += 1

run_concept_drift_and_retraining()


if __name__ == "__main__":
    metrics = train_baseline_model()
    print_metrics(metrics, title="Baseline Random Forest (No Drift)")

import pandas as pd
import os

os.makedirs("results", exist_ok=True)

df = pd.DataFrame({
    "batch": range(len(batch_accuracies)),
    "accuracy": batch_accuracies,
    "before_retrain": acc_before_list,
    "after_retrain": acc_after_list,
    "ks_drift": ks_flags,
    "psi_drift": psi_flags,
    "error_drift": error_flags
})

df.to_csv("results/drift_results.csv", index=False)

print("Results saved to results/drift_results.csv")
