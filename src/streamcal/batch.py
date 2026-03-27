"""
Batch calibrators that refit on accumulated data.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from streamcal.calibrators import BaseCalibrator

EPS = 1e-15


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling calibrator (batch refit every batch)."""

    def __init__(self) -> None:
        self.temperature = 1.0
        self.all_logits: list[NDArray[np.floating[Any]]] = []
        self.all_y: list[NDArray[np.floating[Any]]] = []

    def reset(self) -> None:
        self.temperature = 1.0
        self.all_logits = []
        self.all_y = []

    def calibrate(
        self, p_raw: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe))
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))  # type: ignore[return-value]

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
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
                best_t = float(t)

        self.temperature = best_t

        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))  # type: ignore[return-value]


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator (batch refit every batch)."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.all_p: list[NDArray[np.floating[Any]]] = []
        self.all_y: list[NDArray[np.floating[Any]]] = []
        self.fitted = False

    def reset(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.all_p = []
        self.all_y = []
        self.fitted = False

    def calibrate(
        self, p_raw: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        if not self.fitted:
            return p_raw.copy()
        return self.iso.predict(p_raw)  # type: ignore[return-value]

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        self.all_p.append(p_raw)
        self.all_y.append(y)

        all_p = np.concatenate(self.all_p)
        all_y = np.concatenate(self.all_y)

        self.iso.fit(all_p, all_y)
        self.fitted = True

        return self.iso.predict(p_raw)  # type: ignore[return-value]


class PlattScaling(BaseCalibrator):
    """Platt scaling calibrator (batch refit every k batches)."""

    def __init__(self, refit_every: int = 1) -> None:
        self.refit_every = refit_every
        self.model = LogisticRegression(solver="lbfgs", max_iter=200)
        self.all_logits: list[NDArray[np.floating[Any]]] = []
        self.all_y: list[NDArray[np.floating[Any]]] = []
        self.batch_count = 0
        self.fitted = False

    def reset(self) -> None:
        self.model = LogisticRegression(solver="lbfgs", max_iter=200)
        self.all_logits = []
        self.all_y = []
        self.batch_count = 0
        self.fitted = False

    def calibrate(
        self, p_raw: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        if not self.fitted:
            return p_raw.copy()
        p_safe = np.clip(p_raw, EPS, 1 - EPS)
        logits = np.log(p_safe / (1 - p_safe)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]  # type: ignore[return-value]

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
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

        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]  # type: ignore[return-value]
