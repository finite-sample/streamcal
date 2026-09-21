"""Shared validation for public probability-calibration inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def positive_integer(value: Any, *, name: str) -> int:
    """Return ``value`` as an integer after rejecting booleans and nonpositives."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def unit_interval(value: Any, *, name: str) -> float:
    """Return a finite scalar in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number between 0 and 1")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def nonnegative_finite(value: Any, *, name: str) -> float:
    """Return a finite, nonnegative scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def probabilities(
    values: ArrayLike,
    *,
    name: str = "probabilities",
) -> NDArray[np.float64]:
    """Validate a nonempty one-dimensional array of probabilities."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be between 0 and 1")
    return array


def outcomes(values: ArrayLike, *, name: str = "outcomes") -> NDArray[np.float64]:
    """Validate a nonempty one-dimensional array of binary outcomes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{name} must contain only 0 or 1")
    return array


def paired_binary_data(
    probability_values: ArrayLike,
    outcome_values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate aligned probability and binary-outcome arrays."""
    probability_array = probabilities(probability_values)
    outcome_array = outcomes(outcome_values)
    if probability_array.size != outcome_array.size:
        raise ValueError("probabilities and outcomes must have the same length")
    return probability_array, outcome_array
