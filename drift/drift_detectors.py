import numpy as np
from scipy.stats import ks_2samp

def ks_drift(reference_batch, current_batch, threshold=0.05):
    """
    KS-test based drift detection
    Returns True if drift detected
    """
    drift_flags = []

    for col in reference_batch.select_dtypes(include="number").columns:
        stat, p_value = ks_2samp(reference_batch[col], current_batch[col])
        drift_flags.append(p_value < threshold)

    return any(drift_flags)


def population_stability_index(expected, actual, bins=10):
    """
    PSI calculation for one feature
    """
    expected_percents, _ = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi
def psi_drift(reference_batch, current_batch, threshold=0.2):
    """
    PSI-based drift detection across numeric features
    """
    psi_scores = []

    for col in reference_batch.select_dtypes(include="number").columns:
        psi = population_stability_index(
            reference_batch[col].values,
            current_batch[col].values
        )
        psi_scores.append(psi)

    return max(psi_scores) > threshold
def error_rate_drift(current_accuracy, baseline_accuracy, threshold=0.15):
    """
    Detects drift based on performance degradation.
    Returns True if accuracy drops more than threshold.
    """
    drop = baseline_accuracy - current_accuracy
    return drop > threshold
