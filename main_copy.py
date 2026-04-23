"""
main_copy.py — CLI version of the enhanced drift monitoring pipeline.
Accepts a CSV path and target column as arguments.
Usage: python main_copy.py --data data/raw/credit.csv --target loan_status --batches 10
"""
import argparse
import pandas as pd
import os
from collections import deque

from training.train_baseline_copy import load_and_prepare_dataset, train_baseline_model
from drift.data_stream_copy import stream_batches
from drift.covariate_drift_copy import apply_covariate_drift
from drift.concept_drift_copy import apply_concept_drift
from drift.drift_detectors_copy import ks_drift, psi_drift, error_rate_drift
from evaluation.batch_evaluation_copy import evaluate_batch
from training.retrain_model_copy import retrain_model
from evaluation.performance_metrics_copy import print_metrics


def run_experiment(data_path, target_column, n_batches=10, drift_strength=0.4,
                   concept_ratio=0.4, window_size=5):
    # Load and prepare
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = load_and_prepare_dataset(df, target_column)

    # Train baseline
    model, baseline_metrics = train_baseline_model(X_train, X_test, y_train, y_test)
    print_metrics(baseline_metrics, title="Baseline Model Performance")

    baseline_accuracy = baseline_metrics["accuracy"]
    drift_start = max(1, n_batches // 3)

    # Storage
    results = []
    window_X = deque(maxlen=window_size)
    window_y = deque(maxlen=window_size)

    reference_batch = X_test.iloc[:len(X_test) // n_batches]

    print(f"\nStreaming {n_batches} batches (drift starts at batch {drift_start})...\n")

    for i, (X_batch, y_batch) in enumerate(stream_batches(X_test, y_test, n_batches)):
        if i >= drift_start:
            X_batch = apply_covariate_drift(X_batch, drift_strength=drift_strength)
            y_batch = apply_concept_drift(y_batch, drift_ratio=concept_ratio)

        metrics_before = evaluate_batch(model, X_batch, y_batch)

        if len(window_X) == window_size:
            X_retrain = pd.concat(list(window_X))
            y_retrain = pd.concat(list(window_y))
            model = retrain_model(X_retrain, y_retrain)

        metrics_after = evaluate_batch(model, X_batch, y_batch)

        ks_detected, ks_pval, _ = ks_drift(reference_batch, X_batch)
        psi_detected, psi_max, _ = psi_drift(reference_batch, X_batch)
        err_detected, err_drop = error_rate_drift(metrics_before["accuracy"], baseline_accuracy)

        results.append({
            "batch": i,
            "accuracy": metrics_before["accuracy"],
            "precision": metrics_before["precision"],
            "recall": metrics_before["recall"],
            "f1_score": metrics_before["f1_score"],
            "acc_before": metrics_before["accuracy"],
            "acc_after": metrics_after["accuracy"],
            "ks_drift": int(ks_detected),
            "psi_drift": int(psi_detected),
            "error_drift": int(err_detected),
        })

        window_X.append(X_batch)
        window_y.append(y_batch)

        print(
            f"Batch {i:3d} | "
            f"Acc: {metrics_before['accuracy']:.4f} → {metrics_after['accuracy']:.4f} | "
            f"KS:{ks_detected} PSI:{psi_detected} ERR:{err_detected}"
        )

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/drift_results.csv", index=False)
    print(f"\nResults saved to results/drift_results.csv")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Drift Monitoring Pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--drift-strength", type=float, default=0.4)
    parser.add_argument("--concept-ratio", type=float, default=0.4)
    parser.add_argument("--window-size", type=int, default=5)

    args = parser.parse_args()
    run_experiment(
        args.data, args.target, args.batches,
        args.drift_strength, args.concept_ratio, args.window_size
    )
