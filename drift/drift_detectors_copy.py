import numpy as np
from scipy.stats import ks_2samp


def ks_drift(reference_batch, current_batch, threshold=0.05):
    """
    KS-test based drift detection.
    Returns (drift_detected: bool, avg_p_value: float, per_feature_results: dict)
    """
    drift_flags = []
    p_values = []
    feature_results = {}

    for col in reference_batch.select_dtypes(include="number").columns:
        stat, p_value = ks_2samp(reference_batch[col], current_batch[col])
        drift_flags.append(p_value < threshold)
        p_values.append(p_value)
        feature_results[col] = {"statistic": stat, "p_value": p_value, "drift": p_value < threshold}

    detected = any(drift_flags)
    avg_p = np.mean(p_values) if p_values else 1.0

    return detected, avg_p, feature_results


def population_stability_index(expected, actual, bins=10):
    """
    PSI calculation for one feature.
    """
    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi


def psi_drift(reference_batch, current_batch, threshold=0.2):
    """
    PSI-based drift detection across numeric features.
    Returns (drift_detected: bool, max_psi: float, per_feature_psi: dict)
    """
    psi_scores = {}

    for col in reference_batch.select_dtypes(include="number").columns:
        psi = population_stability_index(
            reference_batch[col].values,
            current_batch[col].values
        )
        psi_scores[col] = psi

    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    detected = max_psi > threshold

    return detected, max_psi, psi_scores


def error_rate_drift(current_accuracy, baseline_accuracy, threshold=0.15):
    """
    Detects drift based on performance degradation.
    Returns (drift_detected: bool, drop: float)
    """
    drop = baseline_accuracy - current_accuracy
    detected = drop > threshold

    return detected, drop
