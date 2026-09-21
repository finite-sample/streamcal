"""Streaming probability calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.isotonic import IsotonicRegression

from streamcal._validation import (
    nonnegative_finite,
    paired_binary_data,
    positive_integer,
    unit_interval,
)
from streamcal._validation import probabilities as validate_probabilities

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


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
        return np.interp(
            probability_array,
            self.bin_centers,
            values,
        )

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
        indices = np.clip(
            np.digitize(probability_array, self.bin_edges) - 1,
            0,
            self.n_bins - 1,
        )

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

        numerator = positive_mass + self.prior_weight * self.bin_centers
        rates = numerator[active] / total_weights[active]
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(
            self.bin_centers[active],
            rates,
            sample_weight=total_weights[active],
        )
        return model.predict(self.bin_centers)
