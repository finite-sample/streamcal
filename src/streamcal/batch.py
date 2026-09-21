"""History and refit policies around scikit-learn's public calibration API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression

from streamcal._validation import paired_binary_data, positive_integer
from streamcal._validation import probabilities as validate_probabilities
from streamcal.calibrators import BaseCalibrator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike, NDArray


class _CalibrationRows:
    """Route all rows to a frozen score adapter without retaining index arrays."""

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: N803, ARG002
        return 1

    def split(self, X, y=None, groups=None) -> Iterator:  # noqa: N803, ARG002
        indices = np.arange(len(X))
        yield indices, indices


class ProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Expose supplied binary probabilities as an already-fitted classifier.

    Sigmoid fits on probabilities; temperature uses scikit-learn's
    log-probability convention. This adapter does not learn a base model.
    """

    def __init__(self) -> None:
        """Declare the fixed binary class order."""
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:  # noqa: N803, ARG002
        """Return this stateless adapter unchanged."""
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:  # noqa: N803
        """Return the most probable binary class."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N803
        """Return class-zero and class-one probabilities in that order."""
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("X must have shape (n_observations, 1)")
        p = validate_probabilities(values[:, 0])
        return np.column_stack((1.0 - p, p))


class BatchCalibrator(BaseCalibrator):
    """Refit an upstream calibrator on an accumulating or rolling history.

    Both-class readiness is required. A one-class window retains the preceding
    fitted map, or identity if no fit has succeeded.
    """

    def __init__(
        self,
        method: str = "isotonic",
        *,
        window_size: int | None = None,
        refit_every: int = 1,
    ) -> None:
        """Configure the upstream method and observation retention policy.

        Args:
            method: One of ``isotonic``, ``sigmoid``, or ``temperature``.
            window_size: Maximum retained observations, or all history if None.
            refit_every: Number of update calls between refits.

        Raises:
            ValueError: If the method is unknown.
        """
        if method not in ("isotonic", "sigmoid", "temperature"):
            raise ValueError("method must be isotonic, sigmoid, or temperature")
        self.method = method
        self.window_size = (
            None
            if window_size is None
            else positive_integer(window_size, name="window_size")
        )
        self.refit_every = positive_integer(refit_every, name="refit_every")
        self.reset()

    def reset(self) -> Self:
        """Discard history and the fitted map."""
        self._probabilities = np.empty(0)
        self._outcomes = np.empty(0)
        self._model: CalibratedClassifierCV | IsotonicRegression | None = None
        self.n_updates = 0
        self.n_observations = 0
        self.frozen = False
        return self

    def freeze(self) -> Self:
        """Keep the fitted map, discard raw history, and ignore future updates."""
        if not self.is_ready:
            raise ValueError("cannot freeze before a successful fit")
        self._probabilities = np.empty(0)
        self._outcomes = np.empty(0)
        self.frozen = True
        return self

    @property
    def is_ready(self) -> bool:
        """Return whether an upstream fit has succeeded."""
        return self._model is not None

    @property
    def history_bytes(self) -> int:
        """Return retained input-array bytes, excluding the fitted model."""
        return self._probabilities.nbytes + self._outcomes.nbytes

    @property
    def retained_observations(self) -> int:
        """Return the number of outcomes available for the next refit."""
        return self._outcomes.size

    def calibrate(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Apply the last completed fit, or identity before readiness."""
        p = validate_probabilities(probabilities)
        if self._model is None:
            return p.copy()
        if isinstance(self._model, IsotonicRegression):
            return self._model.predict(p)
        return self._model.predict_proba(p[:, None])[:, 1]

    def update(self, probabilities: ArrayLike, outcomes: ArrayLike) -> Self:
        """Retain new observations and refit at the configured cadence."""
        p, y = paired_binary_data(probabilities, outcomes)
        if self.frozen:
            return self
        retained_p = np.concatenate((self._probabilities, p))
        retained_y = np.concatenate((self._outcomes, y))
        if self.window_size is not None:
            retained_p = retained_p[-self.window_size :].copy()
            retained_y = retained_y[-self.window_size :].copy()
        model = self._model
        if (self.n_updates + 1) % self.refit_every == 0 and np.unique(
            retained_y
        ).size == 2:
            if self.method == "isotonic":
                model = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                ).fit(retained_p, retained_y)
            else:
                model = CalibratedClassifierCV(
                    FrozenEstimator(ProbabilityClassifier()),
                    method=self.method,
                    cv=_CalibrationRows(),
                ).fit(retained_p[:, None], retained_y)
        self._probabilities, self._outcomes = retained_p, retained_y
        self._model = model
        self.n_updates += 1
        self.n_observations += p.size
        return self
