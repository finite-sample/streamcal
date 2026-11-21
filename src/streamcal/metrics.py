"""Validated diagnostics for binary probability forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    return float(np.mean(np.square(outcome_array - probability_array)))


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
    # Each bin's count cancels the denominator of its mean error.
    bin_errors = np.bincount(indices, weights=outcome_array - probability_array)
    return float(np.sum(np.abs(bin_errors)) / probability_array.size)
