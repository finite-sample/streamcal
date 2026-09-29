"""Streaming probability calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from streamcal._validation import (
    nonnegative_finite,
    paired_binary_data,
    positive_integer,
    unit_interval,
)
from streamcal._validation import probabilities as validate_probabilities

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _grid_index(values: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray:
    """Locate ``values`` on an evenly spaced grid without a binary search.

    Returns ``j`` with ``grid[j] <= value < grid[j + 1]``, clipped to
    ``[0, grid.size - 2]``: exactly what ``np.searchsorted(grid, values,
    "right") - 1`` gives after clipping. The spacing makes ``j`` computable
    directly; one comparison with each neighbour then corrects the rounding
    of that computation, so the result never depends on it. That replaces
    ``O(log n)`` comparisons per value with ``O(1)``.

    Args:
        values: Points to locate, each within one grid spacing of ``grid``.
        grid: Evenly spaced, increasing grid of at least two points.

    Returns:
        Interval index of each value.
    """
    last = grid.size - 2
    index = (values - grid[0]) * ((grid.size - 1) / (grid[-1] - grid[0]))
    index = np.clip(index, 0, last).astype(np.intp)
    index -= (values < grid[index]) & (index > 0)
    index += (values >= grid[index + 1]) & (index < last)
    return index


def _grid_interp(
    values: NDArray[np.float64], grid: NDArray[np.float64], fitted: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``np.interp(values, grid, fitted)`` for an evenly spaced ``grid``.

    Finds each interval with :func:`_grid_index` instead of a binary search,
    then evaluates the same expression ``np.interp`` does, so the result is
    the same to the last bit.

    Args:
        values: Points to interpolate at.
        grid: Evenly spaced, increasing grid of at least two points.
        fitted: Values at the grid points.

    Returns:
        Interpolated values, held constant beyond either end of the grid.
    """
    index = _grid_index(values, grid)
    slopes = (fitted[1:] - fitted[:-1]) / (grid[1:] - grid[:-1])
    result = slopes[index] * (values - grid[index]) + fitted[index]
    result[values < grid[0]] = fitted[0]
    result[values >= grid[-1]] = fitted[-1]
    return result


# Batch sizes from which the grid kernels beat NumPy's binary searches. Below
# them, NumPy's lower fixed cost wins. Both paths give identical results, so
# the switch changes speed only. Measured with ``benchmarks.speed``.
_GRID_INDEX_FROM = 1024
_GRID_INTERP_FROM = 2048


def _bin_index(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray:
    """Bin of each value: ``clip(digitize(values, edges) - 1, 0, n_bins - 1)``."""
    if values.size >= _GRID_INDEX_FROM:
        return _grid_index(values, edges)
    return np.clip(np.digitize(values, edges) - 1, 0, edges.size - 2)


def _interpolate(
    values: NDArray[np.float64], grid: NDArray[np.float64], fitted: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``np.interp(values, grid, fitted)``, using the grid kernel for large inputs."""
    if values.size >= _GRID_INTERP_FROM and grid.size >= 2:
        return _grid_interp(values, grid, fitted)
    return np.interp(values, grid, fitted)


@dataclass(frozen=True)
class CalibratorDiagnostics:
    """A snapshot of the state carried by a streaming calibrator.

    Attributes:
        is_ready: Whether at least one outcome batch has been observed.
        n_updates: Number of calls to :meth:`StreamingIsotonicCalibrator.update`.
        n_observations: Total number of outcomes observed without decay.
        effective_observations: Sum of the decayed bin weights.
        occupied_bins: Number of bins with positive effective weight.
        state_bytes: Bytes used by the fixed-size numerical state arrays.
    """

    is_ready: bool
    n_updates: int
    n_observations: int
    effective_observations: float
    occupied_bins: int
    state_bytes: int


class BaseCalibrator(ABC):
    """Common predict-then-observe contract for binary calibrators."""

    @abstractmethod
    def calibrate(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Calibrate probabilities using only previously observed outcomes."""

    @abstractmethod
    def update(self, probabilities: ArrayLike, outcomes: ArrayLike) -> Self:
        """Observe outcomes and update state without returning predictions."""

    @abstractmethod
    def reset(self) -> Self:
        """Forget all observations and restore identity calibration."""


class StreamingIsotonicCalibrator(BaseCalibrator):
    """Bounded-memory isotonic calibration for probability streams.

    Each update exponentially decays the previous per-bin sufficient statistics,
    adds the new batch, and computes a weighted isotonic fit. The state therefore
    has size ``O(n_bins)`` regardless of stream length.
    """

    def __init__(
        self,
        n_bins: int = 100,
        decay: float | None = None,
        prior_weight: float = 1.0,
        *,
        half_life_seconds: float | None = None,
    ) -> None:
        """Initialize fixed-size sufficient statistics and an identity map.

        Args:
            n_bins: Number of equal-width bins over the probability interval.
            decay: Fraction of earlier effective observations retained per update.
                Set to ``1`` for no forgetting. Defaults to 0.9 when no half-life
                is supplied. Cannot be combined with a half-life.
            prior_weight: Fixed identity-prior weight contributed at each bin
                center. Set to ``0`` to disable the prior.
            half_life_seconds: Positive half-life measured from prediction time.
                Requires explicit prediction and arrival timestamps on updates.

        Raises:
            ValueError: If both forgetting modes are supplied or half-life is zero.
        """
        self.n_bins = positive_integer(n_bins, name="n_bins")
        if half_life_seconds is not None and decay is not None:
            raise ValueError("decay and half_life_seconds are mutually exclusive")
        self.half_life_seconds = half_life_seconds
        if half_life_seconds is not None:
            self.half_life_seconds = nonnegative_finite(
                half_life_seconds, name="half_life_seconds"
            )
            if self.half_life_seconds == 0:
                raise ValueError("half_life_seconds must be positive")
        self.decay = unit_interval(0.9 if decay is None else decay, name="decay")
        self.last_observed_at: float | None = None
        self.prior_weight = nonnegative_finite(prior_weight, name="prior_weight")
        self.bin_edges: NDArray[np.float64] = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_centers: NDArray[np.float64] = (
            self.bin_edges[:-1] + self.bin_edges[1:]
        ) / 2.0
        self.effective_counts: NDArray[np.float64] = np.zeros(self.n_bins)
        self.positive_mass: NDArray[np.float64] = np.zeros(self.n_bins)
        self.calibration_values: NDArray[np.float64] = self.bin_centers.copy()
        self.n_updates = 0
        self.n_observations = 0

    def reset(self) -> Self:
        """Forget all observations and restore identity calibration."""
        self.last_observed_at = None
        self.effective_counts.fill(0.0)
        self.positive_mass.fill(0.0)
        self.calibration_values = self.bin_centers.copy()
        self.n_updates = 0
        self.n_observations = 0
        return self

    def calibrate(
        self, probabilities: ArrayLike, *, current_time: float | None = None
    ) -> NDArray[np.float64]:
        """Calibrate without mutating learned state.

        Args:
            probabilities: Binary forecast probabilities.
            current_time: Finite seconds on the update timestamp clock. Required
                in half-life mode and must not precede the latest arrival.

        Returns:
            Calibrated probabilities in input order.
        """
        probability_array = validate_probabilities(probabilities)
        factor = self._time_factor(current_time)
        if self.n_updates == 0:
            return probability_array.copy()
        values = self.calibration_values
        if self.half_life_seconds is not None:
            values = self._fit_values(
                self.effective_counts * factor, self.positive_mass * factor
            )
        return _interpolate(probability_array, self.bin_centers, values)

    def update(
        self,
        probabilities: ArrayLike,
        outcomes: ArrayLike,
        *,
        prediction_times: ArrayLike | None = None,
        observed_at: float | None = None,
    ) -> Self:
        """Observe labels, weighting late outcomes by their prediction age.

        Args:
            probabilities: Original forecast probabilities, not calibrated ones.
            outcomes: Corresponding binary labels.
            prediction_times: One finite prediction timestamp per observation.
            observed_at: Finite, nondecreasing label arrival timestamp in seconds.

        Returns:
            This calibrator after the update.

        Raises:
            ValueError: If timing arguments are missing, misaligned or inconsistent.
        """
        probability_array, outcome_array = paired_binary_data(probabilities, outcomes)
        factor = self._time_factor(observed_at)
        weights = np.ones(probability_array.size)
        if self.half_life_seconds is not None:
            if observed_at is None:
                raise ValueError("observed_at is required in half-life mode")
            times = np.asarray(prediction_times, dtype=float)
            if times.shape != probability_array.shape or not np.all(np.isfinite(times)):
                raise ValueError(
                    "prediction_times must be finite and aligned with probabilities"
                )
            if np.any(times > observed_at):
                raise ValueError("prediction_times cannot follow observed_at")
            weights = np.exp2(-(float(observed_at) - times) / self.half_life_seconds)
        elif prediction_times is not None:
            raise ValueError("prediction_times requires half_life_seconds")
        indices = _bin_index(probability_array, self.bin_edges)

        self.effective_counts *= factor
        self.positive_mass *= factor
        self.effective_counts += np.bincount(
            indices, weights=weights, minlength=self.n_bins
        )
        self.positive_mass += np.bincount(
            indices,
            weights=outcome_array * weights,
            minlength=self.n_bins,
        )
        self.n_updates += 1
        self.n_observations += probability_array.size
        self.last_observed_at = observed_at
        self.calibration_values = self._fit_values(
            self.effective_counts, self.positive_mass
        )
        return self

    def diagnostics(self) -> CalibratorDiagnostics:
        """Return readiness, data-use, and fixed-state memory diagnostics."""
        state_bytes = sum(
            array.nbytes
            for array in (
                self.bin_edges,
                self.bin_centers,
                self.effective_counts,
                self.positive_mass,
                self.calibration_values,
            )
        )
        return CalibratorDiagnostics(
            is_ready=self.n_updates > 0,
            n_updates=self.n_updates,
            n_observations=self.n_observations,
            effective_observations=float(self.effective_counts.sum()),
            occupied_bins=int(np.count_nonzero(self.effective_counts > 0.0)),
            state_bytes=state_bytes,
        )

    def _time_factor(self, timestamp: float | None) -> float:
        if self.half_life_seconds is None:
            if timestamp is not None:
                raise ValueError("timestamps require half_life_seconds")
            return self.decay
        if (
            timestamp is None
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float, np.integer, np.floating))
        ):
            raise ValueError("a finite timestamp is required in half-life mode")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.last_observed_at is None:
            return 1.0
        if timestamp < self.last_observed_at:
            raise ValueError("timestamp must not precede the latest observed_at")
        return float(
            np.exp2(-(timestamp - self.last_observed_at) / self.half_life_seconds)
        )

    def _fit_values(
        self, counts: NDArray[np.float64], positive_mass: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        total_weights = counts + self.prior_weight
        active = total_weights > 0.0
        if not np.any(active):
            return self.bin_centers.copy()

        # Imported here so that importing streamcal does not load scipy.optimize.
        from scipy.optimize import isotonic_regression

        numerator = positive_mass + self.prior_weight * self.bin_centers
        rates = numerator[active] / total_weights[active]
        fitted = isotonic_regression(rates, weights=total_weights[active]).x
        np.clip(fitted, 0.0, 1.0, out=fitted)
        if active.all():
            return fitted
        # Bins with no weight take values interpolated from their neighbours.
        return np.interp(self.bin_centers, self.bin_centers[active], fitted)
