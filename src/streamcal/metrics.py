"""Validated diagnostics for binary probability forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from streamcal._validation import paired_binary_data, positive_integer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def brier_score(outcomes: ArrayLike, probabilities: ArrayLike) -> float:
    """Return mean squared probability error for binary outcomes.

    Args:
        outcomes: Observed binary outcomes.
        probabilities: Forecast probabilities aligned with ``outcomes``.

    Returns:
        Mean squared probability error.
    """
    probability_array, outcome_array = paired_binary_data(probabilities, outcomes)
    return float(brier_score_loss(outcome_array, probability_array, pos_label=1))


def binned_calibration_error(
    outcomes: ArrayLike,
    probabilities: ArrayLike,
    n_bins: int = 20,
) -> float:
    """Return equal-width binned absolute calibration error.

    This diagnostic depends on both sample size and ``n_bins``. It is useful for
    visualization and comparison at a fixed design, but it is not a proper score
    and should not be used alone to select a probabilistic forecast.

    Args:
        outcomes: Observed binary outcomes.
        probabilities: Forecast probabilities aligned with ``outcomes``.
        n_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Observation-weighted absolute gap between bin means.
    """
    n_bins = positive_integer(n_bins, name="n_bins")
    probability_array, outcome_array = paired_binary_data(probabilities, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.searchsorted(edges[1:-1], probability_array)
    counts = np.bincount(indices, minlength=n_bins)
    observed, predicted = calibration_curve(
        outcome_array, probability_array, n_bins=n_bins, pos_label=1
    )
    return float(np.average(np.abs(observed - predicted), weights=counts[counts > 0]))
