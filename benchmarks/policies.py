"""Benchmark adapters; numerical learning remains in upstream packages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from river import linear_model, optim
from scipy.special import logit

from streamcal._validation import paired_binary_data
from streamcal._validation import probabilities as validate_probabilities
from streamcal.calibrators import BaseCalibrator

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class RawReference(BaseCalibrator):
    """Identity predictions with no retained observations."""

    def calibrate(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Return unchanged validated probabilities."""
        return validate_probabilities(probabilities).copy()

    def update(self, probabilities: ArrayLike, outcomes: ArrayLike) -> Self:
        """Validate inputs without learning from them."""
        paired_binary_data(probabilities, outcomes)
        return self

    def reset(self) -> Self:
        """Return this stateless reference."""
        return self


class OnlineLogisticReference(BaseCalibrator):
    """River logistic regression on raw probability logits."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        """Configure River's SGD learning rate."""
        self.learning_rate = learning_rate
        self.reset()

    def reset(self) -> Self:
        """Reset the upstream learner and observation count."""
        self.model = linear_model.LogisticRegression(
            optimizer=optim.SGD(self.learning_rate),
            intercept_lr=self.learning_rate,
        )
        self.n_observations = 0
        return self

    @staticmethod
    def _features(p: float) -> dict[str, float]:
        epsilon = np.finfo(float).eps
        return {"logit": float(logit(np.clip(p, epsilon, 1 - epsilon)))}

    def calibrate(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Predict using River, or identity before any observation."""
        p = validate_probabilities(probabilities)
        if self.n_observations == 0:
            return p.copy()
        return np.asarray(
            [self.model.predict_proba_one(self._features(v))[True] for v in p]
        )

    def update(self, probabilities: ArrayLike, outcomes: ArrayLike) -> Self:
        """Update River one observation at a time in arrival order."""
        p, y = paired_binary_data(probabilities, outcomes)
        for value, label in zip(p, y, strict=True):
            self.model.learn_one(self._features(value), bool(label))
        self.n_observations += p.size
        return self
