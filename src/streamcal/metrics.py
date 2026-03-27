"""
Calibration metrics.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def brier_score(
    y_true: NDArray[np.floating[Any]], y_pred: NDArray[np.floating[Any]]
) -> float:
    """Brier score: mean squared error between predictions and outcomes."""
    return float(np.mean((y_pred - y_true) ** 2))


def expected_calibration_error(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    n_bins: int = 20,
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

    return float(total_ece)
