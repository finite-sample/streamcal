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


def _drop_head[T: np.generic](chunk: NDArray[T], count: int) -> NDArray[T]:
    """Drop the first ``count`` rows of a history chunk.

    A slice keeps its whole parent array alive, so once the kept part is under
    half of the parent it is copied and the dropped rows are freed.
    """
    kept = chunk[count:]
    if kept.base is not None and 2 * kept.size < kept.base.size:
        kept = kept.copy()
    return kept


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
        self._clear_history()
        self._model: CalibratedClassifierCV | IsotonicRegression | None = None
        self.n_updates = 0
        self.n_observations = 0
        self.frozen = False
        return self

    def freeze(self) -> Self:
        """Keep the fitted map, discard raw history, and ignore future updates."""
        if not self.is_ready:
            raise ValueError("cannot freeze before a successful fit")
        self._clear_history()
        self.frozen = True
        return self

    def _clear_history(self) -> None:
        # History is kept as a list of per-update chunks, joined only when a
        # refit needs it, so an update that does not refit costs O(batch)
        # rather than copying all history. Labels are stored as one byte each.
        self._probability_chunks: list[NDArray[np.float64]] = []
        self._outcome_chunks: list[NDArray[np.uint8]] = []
        self._retained = 0

    def _append(self, p: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self._probability_chunks.append(p.copy())
        self._outcome_chunks.append(y.astype(np.uint8))
        self._retained += p.size
        excess = 0 if self.window_size is None else self._retained - self.window_size
        while excess > 0:
            oldest = self._probability_chunks[0].size
            if oldest <= excess:
                del self._probability_chunks[0], self._outcome_chunks[0]
                self._retained -= oldest
                excess -= oldest
                continue
            self._probability_chunks[0] = _drop_head(
                self._probability_chunks[0], excess
            )
            self._outcome_chunks[0] = _drop_head(self._outcome_chunks[0], excess)
            self._retained -= excess
            excess = 0

    def _history(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(self._probability_chunks) > 1:
            self._probability_chunks = [np.concatenate(self._probability_chunks)]
            self._outcome_chunks = [np.concatenate(self._outcome_chunks)]
        if not self._probability_chunks:
            return np.empty(0), np.empty(0)
        return (
            self._probability_chunks[0],
            self._outcome_chunks[0].astype(np.float64),
        )

    @property
    def is_ready(self) -> bool:
        """Return whether an upstream fit has succeeded."""
        return self._model is not None

    @property
    def history_bytes(self) -> int:
        """Return retained input-array bytes, excluding the fitted model."""
        return sum(
            chunk.nbytes for chunk in self._probability_chunks + self._outcome_chunks
        )

    @property
    def retained_observations(self) -> int:
        """Return the number of outcomes available for the next refit."""
        return self._retained

    def calibrate(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Apply the last completed fit, or identity before readiness."""
        p = validate_probabilities(probabilities)
        if self._model is None:
            return p.copy()
        if isinstance(self._model, IsotonicRegression):
            return self._model.predict(p)
        return self._model.predict_proba(p[:, None])[:, 1]

    def _fit(
        self, p: NDArray[np.float64], y: NDArray[np.float64]
    ) -> CalibratedClassifierCV | IsotonicRegression:
        if self.method == "isotonic":
            return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                p, y
            )
        return CalibratedClassifierCV(
            FrozenEstimator(ProbabilityClassifier()),
            method=self.method,
            cv=_CalibrationRows(),
        ).fit(p[:, None], y)

    def update(self, probabilities: ArrayLike, outcomes: ArrayLike) -> Self:
        """Retain new observations and refit at the configured cadence."""
        p, y = paired_binary_data(probabilities, outcomes)
        if self.frozen:
            return self
        self._append(p, y)
        if (self.n_updates + 1) % self.refit_every == 0:
            retained_p, retained_y = self._history()
            if np.unique(retained_y).size == 2:
                self._model = self._fit(retained_p, retained_y)
        self.n_updates += 1
        self.n_observations += p.size
        return self
