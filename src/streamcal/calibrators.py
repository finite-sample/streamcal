"""
Streaming calibrators for probability calibration.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


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


class PerBucketEMA(BaseCalibrator):
    """Per-bucket exponential moving average calibrator."""

    def __init__(self, n_buckets: int = 100, alpha: float = 0.3) -> None:
        self.n_buckets = n_buckets
        self.alpha = alpha
        self.bins: NDArray[np.floating[Any]] = np.linspace(0, 1, n_buckets + 1)
        self.ema_rates: NDArray[np.floating[Any]] = np.full(n_buckets, 0.5)
        self.has_seen: NDArray[np.bool_] = np.zeros(n_buckets, dtype=bool)

    def reset(self) -> None:
        self.ema_rates = np.full(self.n_buckets, 0.5)
        self.has_seen = np.zeros(self.n_buckets, dtype=bool)

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        return self.ema_rates[bucket_idx]  # type: ignore[return-value]

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)

        outcome_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)
        mask = counts > 0

        batch_rates: NDArray[np.floating[Any]] = np.zeros(self.n_buckets)
        batch_rates[mask] = outcome_sum[mask] / counts[mask]

        for b in range(self.n_buckets):
            if counts[b] > 0:
                if not self.has_seen[b]:
                    self.ema_rates[b] = batch_rates[b]
                    self.has_seen[b] = True
                else:
                    self.ema_rates[b] = (1 - self.alpha) * self.ema_rates[
                        b
                    ] + self.alpha * batch_rates[b]

        return self.ema_rates[bucket_idx]  # type: ignore[return-value]


class OnlineSGD(BaseCalibrator):
    """Online SGD calibrator with additive updates."""

    def __init__(
        self,
        n_buckets: int = 100,
        eta: float = 0.1,
        c_min: float = 0.1,
        c_max: float = 10.0,
    ) -> None:
        self.n_buckets = n_buckets
        self.eta = eta
        self.c_min = c_min
        self.c_max = c_max
        self.bins: NDArray[np.floating[Any]] = np.linspace(0, 1, n_buckets + 1)
        self.theta: NDArray[np.floating[Any]] = np.zeros(n_buckets)

    def reset(self) -> None:
        self.theta = np.zeros(self.n_buckets)

    def _odds_calibrate(
        self, p_raw: NDArray[np.floating[Any]], theta: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        c = np.exp(theta)
        return (c * p_raw) / (1 - p_raw + c * p_raw)  # type: ignore[return-value]

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        return self._odds_calibrate(p_raw, self.theta[bucket_idx])

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        p_cal = self._odds_calibrate(p_raw, self.theta[bucket_idx])

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors: NDArray[np.floating[Any]] = np.zeros(self.n_buckets)
        errors[mask] = pred_sum[mask] / counts[mask] - true_sum[mask] / counts[mask]

        self.theta -= self.eta * errors
        self.theta = np.clip(self.theta, np.log(self.c_min), np.log(self.c_max))

        return p_cal


class MWUCalibrator(BaseCalibrator):
    """Multiplicative Weights Update calibrator."""

    def __init__(
        self,
        n_buckets: int = 100,
        eta: float = 0.1,
        c_min: float = 0.1,
        c_max: float = 10.0,
        diminishing_lr: bool = False,
        mu: float = 0.1,
    ) -> None:
        self.n_buckets = n_buckets
        self.eta = eta
        self.c_min = c_min
        self.c_max = c_max
        self.diminishing_lr = diminishing_lr
        self.mu = mu
        self.bins: NDArray[np.floating[Any]] = np.linspace(0, 1, n_buckets + 1)
        self.weights: NDArray[np.floating[Any]] = np.ones(n_buckets)
        self.t = 0

    def reset(self) -> None:
        self.weights = np.ones(self.n_buckets)
        self.t = 0

    def calibrate(self, p_raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        return np.clip(p_cal, 0, 1)  # type: ignore[return-value]

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        self.t += 1
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        p_cal = np.clip(p_cal, 0, 1)

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors: NDArray[np.floating[Any]] = np.zeros(self.n_buckets)
        errors[mask] = pred_sum[mask] / counts[mask] - true_sum[mask] / counts[mask]

        if self.diminishing_lr:
            eta_t = self.eta / (self.mu * self.t)
        else:
            eta_t = self.eta

        self.weights *= np.exp(-eta_t * errors)
        self.weights = np.clip(self.weights, self.c_min, self.c_max)

        return p_cal  # type: ignore[return-value]

    def get_bucket_errors(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
        """Get bucket errors without updating (for metrics)."""
        bucket_idx = np.clip(np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1)
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        p_cal = np.clip(p_cal, 0, 1)

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors: NDArray[np.floating[Any]] = np.zeros(self.n_buckets)
        errors[mask] = pred_sum[mask] / counts[mask] - true_sum[mask] / counts[mask]

        return errors, counts
