"""
Calibration methods for experiments.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

EPS = 1e-15


class BaseCalibrator:
    """Base class for calibrators."""

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        raise NotImplementedError

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calibrate and update based on outcomes."""
        raise NotImplementedError

    def reset(self):
        """Reset calibrator state."""
        raise NotImplementedError


class NoCalibration(BaseCalibrator):
    """No calibration - returns raw probabilities unchanged."""

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        return p_raw.copy()

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        return p_raw.copy()

    def reset(self):
        pass


class PerBucketEMA(BaseCalibrator):
    """Per-bucket exponential moving average calibrator."""

    def __init__(self, n_buckets: int = 100, alpha: float = 0.3):
        self.n_buckets = n_buckets
        self.alpha = alpha
        self.bins = np.linspace(0, 1, n_buckets + 1)
        self.ema_rates = np.full(n_buckets, 0.5)
        self.has_seen = np.zeros(n_buckets, dtype=bool)

    def reset(self):
        self.ema_rates = np.full(self.n_buckets, 0.5)
        self.has_seen = np.zeros(self.n_buckets, dtype=bool)

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        return self.ema_rates[bucket_idx]

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )

        outcome_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)
        mask = counts > 0

        batch_rates = np.zeros(self.n_buckets)
        batch_rates[mask] = outcome_sum[mask] / counts[mask]

        for b in range(self.n_buckets):
            if counts[b] > 0:
                if not self.has_seen[b]:
                    self.ema_rates[b] = batch_rates[b]
                    self.has_seen[b] = True
                else:
                    self.ema_rates[b] = (
                        (1 - self.alpha) * self.ema_rates[b]
                        + self.alpha * batch_rates[b]
                    )

        return self.ema_rates[bucket_idx]


class OnlineSGD(BaseCalibrator):
    """Online SGD calibrator with additive updates."""

    def __init__(
        self,
        n_buckets: int = 100,
        eta: float = 0.1,
        c_min: float = 0.1,
        c_max: float = 10.0,
    ):
        self.n_buckets = n_buckets
        self.eta = eta
        self.c_min = c_min
        self.c_max = c_max
        self.bins = np.linspace(0, 1, n_buckets + 1)
        self.theta = np.zeros(n_buckets)

    def reset(self):
        self.theta = np.zeros(self.n_buckets)

    def _odds_calibrate(self, p_raw: np.ndarray, theta: np.ndarray) -> np.ndarray:
        c = np.exp(theta)
        return (c * p_raw) / (1 - p_raw + c * p_raw)

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        return self._odds_calibrate(p_raw, self.theta[bucket_idx])

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        p_cal = self._odds_calibrate(p_raw, self.theta[bucket_idx])

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors = np.zeros(self.n_buckets)
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
    ):
        self.n_buckets = n_buckets
        self.eta = eta
        self.c_min = c_min
        self.c_max = c_max
        self.diminishing_lr = diminishing_lr
        self.mu = mu
        self.bins = np.linspace(0, 1, n_buckets + 1)
        self.weights = np.ones(n_buckets)
        self.t = 0

    def reset(self):
        self.weights = np.ones(self.n_buckets)
        self.t = 0

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        return np.clip(p_cal, 0, 1)

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.t += 1
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        p_cal = np.clip(p_cal, 0, 1)

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors = np.zeros(self.n_buckets)
        errors[mask] = pred_sum[mask] / counts[mask] - true_sum[mask] / counts[mask]

        if self.diminishing_lr:
            eta_t = self.eta / (self.mu * self.t)
        else:
            eta_t = self.eta

        self.weights *= np.exp(-eta_t * errors)
        self.weights = np.clip(self.weights, self.c_min, self.c_max)

        return p_cal

    def get_bucket_errors(
        self, p_raw: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get bucket errors without updating (for metrics)."""
        bucket_idx = np.clip(
            np.digitize(p_raw, self.bins) - 1, 0, self.n_buckets - 1
        )
        bias = self.weights[bucket_idx]
        p_cal = (bias * p_raw) / (1 - p_raw + bias * p_raw)
        p_cal = np.clip(p_cal, 0, 1)

        pred_sum = np.bincount(bucket_idx, weights=p_cal, minlength=self.n_buckets)
        true_sum = np.bincount(bucket_idx, weights=y, minlength=self.n_buckets)
        counts = np.bincount(bucket_idx, minlength=self.n_buckets)

        mask = counts > 0
        errors = np.zeros(self.n_buckets)
        errors[mask] = pred_sum[mask] / counts[mask] - true_sum[mask] / counts[mask]

        return errors, counts


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling calibrator (batch refit every batch)."""

    def __init__(self):
        self.temperature = 1.0
        self.all_logits = []
        self.all_y = []

    def reset(self):
        self.temperature = 1.0
        self.all_logits = []
        self.all_y = []

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe))
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe))
        self.all_logits.append(logits)
        self.all_y.append(y)

        all_logits = np.concatenate(self.all_logits)
        all_y = np.concatenate(self.all_y)

        best_t = 1.0
        best_loss = float("inf")
        for t in np.logspace(-1, 1, 50):
            probs = 1 / (1 + np.exp(-all_logits / t))
            probs = np.clip(probs, EPS, 1 - EPS)
            loss = -np.mean(all_y * np.log(probs) + (1 - all_y) * np.log(1 - probs))
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = best_t

        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator (batch refit every batch)."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.all_p = []
        self.all_y = []
        self.fitted = False

    def reset(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.all_p = []
        self.all_y = []
        self.fitted = False

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p_raw.copy()
        return self.iso.predict(p_raw)

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.all_p.append(p_raw)
        self.all_y.append(y)

        all_p = np.concatenate(self.all_p)
        all_y = np.concatenate(self.all_y)

        self.iso.fit(all_p, all_y)
        self.fitted = True

        return self.iso.predict(p_raw)


class PlattScaling(BaseCalibrator):
    """Platt scaling calibrator (batch refit every k batches)."""

    def __init__(self, refit_every: int = 1):
        self.refit_every = refit_every
        self.model = LogisticRegression(solver="lbfgs", max_iter=200)
        self.all_logits = []
        self.all_y = []
        self.batch_count = 0
        self.fitted = False

    def reset(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=200)
        self.all_logits = []
        self.all_y = []
        self.batch_count = 0
        self.fitted = False

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p_raw.copy()
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

    def update(self, p_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.batch_count += 1
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe))
        self.all_logits.append(logits)
        self.all_y.append(y)

        if self.batch_count % self.refit_every == 0:
            all_logits = np.concatenate(self.all_logits).reshape(-1, 1)
            all_y = np.concatenate(self.all_y)
            self.model.fit(all_logits, all_y)
            self.fitted = True

        if not self.fitted:
            return p_raw.copy()

        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]


def create_calibrator(method: str, **kwargs) -> BaseCalibrator:
    """Factory function for calibrators."""
    calibrators = {
        "none": NoCalibration,
        "ema": PerBucketEMA,
        "sgd": OnlineSGD,
        "mwu": MWUCalibrator,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
        "platt": PlattScaling,
    }

    if method not in calibrators:
        raise ValueError(f"Unknown method: {method}. Choose from {list(calibrators.keys())}")

    return calibrators[method](**kwargs)
