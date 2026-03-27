"""
Metrics for calibration experiments.
"""

import numpy as np


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score: mean squared error between predictions and outcomes."""
    return np.mean((y_pred - y_true) ** 2)


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 20
) -> float:
    """
    Expected Calibration Error (ECE).

    Weighted average of absolute calibration error across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)

    total_ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            bin_weight = mask.sum() / n
            total_ece += bin_weight * np.abs(bin_acc - bin_conf)

    return total_ece


def per_bucket_mse(
    bucket_errors: np.ndarray, bucket_counts: np.ndarray
) -> float:
    """
    Per-bucket mean squared error: (1/B) * sum(ell_b^2) for non-empty buckets.

    Args:
        bucket_errors: Array of calibration errors per bucket
        bucket_counts: Array of counts per bucket

    Returns:
        Mean squared calibration error across buckets with observations
    """
    mask = bucket_counts > 0
    if not mask.any():
        return 0.0
    errors = bucket_errors[mask]
    return np.mean(errors ** 2)


def compute_bucket_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_idx: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-bucket calibration errors.

    Args:
        y_true: Binary outcomes
        y_pred: Calibrated probabilities
        bucket_idx: Bucket assignment for each observation
        n_buckets: Number of buckets

    Returns:
        Tuple of (bucket_errors, bucket_counts)
    """
    bucket_pred_sum = np.bincount(bucket_idx, weights=y_pred, minlength=n_buckets)
    bucket_true_sum = np.bincount(bucket_idx, weights=y_true, minlength=n_buckets)
    bucket_counts = np.bincount(bucket_idx, minlength=n_buckets)

    bucket_errors = np.zeros(n_buckets)
    mask = bucket_counts > 0
    bucket_errors[mask] = (
        bucket_pred_sum[mask] / bucket_counts[mask]
        - bucket_true_sum[mask] / bucket_counts[mask]
    )

    return bucket_errors, bucket_counts


def time_to_recovery(
    ece_series: np.ndarray,
    shift_batch: int,
    threshold_factor: float = 1.5,
) -> int:
    """
    Compute batches until ECE returns to threshold after a shift.

    Args:
        ece_series: ECE values per batch
        shift_batch: Batch index where shift occurred
        threshold_factor: Recovery threshold as multiple of pre-shift ECE

    Returns:
        Number of batches to recover, or -1 if never recovered
    """
    if shift_batch <= 0:
        return -1

    pre_shift_ece = np.mean(ece_series[:shift_batch])
    threshold = threshold_factor * pre_shift_ece

    for i in range(shift_batch, len(ece_series)):
        if ece_series[i] <= threshold:
            return i - shift_batch

    return -1


def compute_convergence_rate(
    mse_series: np.ndarray, start_fraction: float = 0.5
) -> float:
    """
    Compute empirical convergence rate from log-log slope.

    Args:
        mse_series: Per-bucket MSE values per batch
        start_fraction: Fraction of series to skip before fitting

    Returns:
        Empirical slope (should be ~-1 for O(1/t) convergence)
    """
    n = len(mse_series)
    start_idx = int(n * start_fraction)
    if start_idx >= n - 2:
        return 0.0

    mse_subset = mse_series[start_idx:]
    mse_subset = mse_subset[mse_subset > 0]
    if len(mse_subset) < 3:
        return 0.0

    t = np.arange(start_idx + 1, start_idx + 1 + len(mse_subset))
    log_t = np.log(t)
    log_mse = np.log(mse_subset)

    slope, _ = np.polyfit(log_t, log_mse, 1)
    return slope
