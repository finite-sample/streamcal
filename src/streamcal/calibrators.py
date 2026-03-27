"""
Streaming calibrators for probability calibration.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]


class BaseCalibrator:
    """Base class for calibrators."""

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Apply calibration to raw probabilities."""
        raise NotImplementedError

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Calibrate and update based on outcomes."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset calibrator state."""
        raise NotImplementedError


class StreamingIsotonicCalibrator(BaseCalibrator):
    """Streaming isotonic calibration with EMA + PAV.

    Maintains exponential moving average of bucket outcome rates,
    then projects onto monotonic cone via PAV. Adapts quickly to
    drift while guaranteeing monotonicity.

    Parameters
    ----------
    n_buckets : int
        Number of buckets for discretizing probability space.
    alpha : float
        EMA smoothing factor. Larger values adapt faster to drift
        but have more variance. Smaller values are more stable.
    """

    def __init__(self, n_buckets: int = 100, alpha: float = 0.1) -> None:
        self.n_buckets = n_buckets
        self.alpha = alpha
        self.bins: NDArray[np.floating[Any]] = np.linspace(0, 1, n_buckets + 1)
        self.bucket_centers: NDArray[np.floating[Any]] = (
            self.bins[:-1] + self.bins[1:]
        ) / 2
        self.ema_rates: NDArray[np.floating[Any]] = self.bucket_centers.copy()
        self.has_seen: NDArray[np.bool_] = np.zeros(n_buckets, dtype=bool)
        self.calibration_values: NDArray[np.floating[Any]] = self.bucket_centers.copy()

    def reset(self) -> None:
        self.ema_rates = self.bucket_centers.copy()
        self.has_seen = np.zeros(self.n_buckets, dtype=bool)
        self.calibration_values = self.bucket_centers.copy()

    def _pav(self, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Project onto monotonic cone via isotonic regression."""
        iso = IsotonicRegression(out_of_bounds="clip")
        return iso.fit_transform(self.bucket_centers, y)  # type: ignore[no-any-return]

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Apply monotonic calibration via interpolation."""
        return np.interp(p_raw, self.bucket_centers, self.calibration_values)

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Update EMA rates and recompute monotonic calibration."""
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)

        outcome_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        for b in range(self.n_buckets):
            if counts[b] > 0:
                batch_rate = outcome_sum[b] / counts[b]
                if not self.has_seen[b]:
                    self.ema_rates[b] = batch_rate
                    self.has_seen[b] = True
                else:
                    self.ema_rates[b] = (1 - self.alpha) * self.ema_rates[
                        b
                    ] + self.alpha * batch_rate

        self.calibration_values = self._pav(self.ema_rates)

        return self.calibrate(p_raw)


class NearlyIsotonicCalibrator(BaseCalibrator):
    """Streaming nearly-isotonic calibration via SGD with penalty.

    Uses Tibshirani et al. (2011) nearly-isotonic penalty:
    L(θ) = Σ (y_b - θ_b)² + λ Σ max(0, θ_b - θ_{b+1})

    This implementation maintains EMA estimates of bucket rates (like
    StreamingIsotonicCalibrator) but uses a soft penalty instead of PAV
    projection to encourage monotonicity.

    Parameters
    ----------
    n_buckets : int
        Number of buckets for discretizing probability space.
    lam : float
        Penalty strength. Larger values enforce monotonicity more strictly.
        λ → ∞ recovers isotonic regression, λ = 0 is unconstrained.
    alpha : float
        EMA smoothing factor for bucket rate estimates.
    eta : float
        Learning rate for SGD monotonicity updates.
    """

    def __init__(
        self,
        n_buckets: int = 100,
        lam: float = 1.0,
        alpha: float = 0.1,
        eta: float = 0.01,
        n_iters: int = 100,
    ) -> None:
        self.n_buckets = n_buckets
        self.lam = lam
        self.alpha = alpha
        self.eta = eta
        self.n_iters = n_iters
        self.bins: NDArray[np.floating[Any]] = np.linspace(0, 1, n_buckets + 1)
        self.bucket_centers: NDArray[np.floating[Any]] = (
            self.bins[:-1] + self.bins[1:]
        ) / 2
        self.ema_rates: NDArray[np.floating[Any]] = self.bucket_centers.copy()
        self.has_seen: NDArray[np.bool_] = np.zeros(n_buckets, dtype=bool)
        self.calibration_values: NDArray[np.floating[Any]] = self.bucket_centers.copy()

    def reset(self) -> None:
        self.ema_rates = self.bucket_centers.copy()
        self.has_seen = np.zeros(self.n_buckets, dtype=bool)
        self.calibration_values = self.bucket_centers.copy()

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Apply calibration via interpolation."""
        return np.interp(p_raw, self.bucket_centers, self.calibration_values)

    def _apply_nearly_isotonic(
        self, rates: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Apply nearly-isotonic penalty via SGD."""
        theta = rates.copy()

        for _ in range(self.n_iters):
            grad = np.zeros(self.n_buckets)

            for b in range(self.n_buckets):
                grad[b] = 2.0 * (theta[b] - rates[b])

                if b < self.n_buckets - 1 and theta[b] > theta[b + 1]:
                    grad[b] += self.lam
                if b > 0 and theta[b - 1] > theta[b]:
                    grad[b] -= self.lam

            theta = theta - self.eta * grad
            theta = np.clip(theta, 0.0, 1.0)

        return theta

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Update EMA rates and apply nearly-isotonic penalty."""
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)

        outcome_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        for b in range(self.n_buckets):
            if counts[b] > 0:
                batch_rate = outcome_sum[b] / counts[b]
                if not self.has_seen[b]:
                    self.ema_rates[b] = batch_rate
                    self.has_seen[b] = True
                else:
                    self.ema_rates[b] = (
                        1 - self.alpha
                    ) * self.ema_rates[b] + self.alpha * batch_rate

        self.calibration_values = self._apply_nearly_isotonic(self.ema_rates)

        return self.calibrate(p_raw)
