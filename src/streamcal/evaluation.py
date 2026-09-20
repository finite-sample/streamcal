"""Leak-free prequential comparison of streaming-isotonic configurations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from streamcal._validation import paired_binary_data, positive_integer
from streamcal.calibrators import BaseCalibrator, StreamingIsotonicCalibrator
from streamcal.metrics import binned_calibration_error, brier_score

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import ArrayLike
    from river.base.typing import Dataset


@dataclass(frozen=True)
class StreamingIsotonicConfig:
    """Configuration used by :func:`compare_prequential`."""

    n_bins: int = 100
    decay: float | None = None
    prior_weight: float = 1.0
    half_life_seconds: float | None = None

    def build(self) -> StreamingIsotonicCalibrator:
        """Construct and validate a fresh calibrator."""
        return StreamingIsotonicCalibrator(
            n_bins=self.n_bins,
            decay=self.decay,
            prior_weight=self.prior_weight,
            half_life_seconds=self.half_life_seconds,
        )


@dataclass(frozen=True)
class ConfigurationResult:
    """Quality and resource measurements for one configuration."""

    name: str
    brier: float
    log_loss: float
    binned_calibration_error: float
    roc_auc: float
    predict_ns_per_observation: float
    update_ns_per_observation: float
    state_bytes: int
    batch_brier: tuple[float, ...]
    cumulative_brier: tuple[float, ...]
    serialized_bytes: int = 0
    squared_errors: tuple[float, ...] = ()
    p95_predict_ms: float = 0.0
    p95_update_ms: float = 0.0


@dataclass(frozen=True)
class TradeoffReport:
    """Prequential results for a configuration comparison."""

    n_observations: int
    batch_size: int
    results: tuple[ConfigurationResult, ...]

    def pareto_frontier(self) -> tuple[ConfigurationResult, ...]:
        """Return nondominated Brier, serialized-size, and update-time choices."""

        def costs(result: ConfigurationResult) -> tuple[float, ...]:
            return (
                result.brier,
                result.serialized_bytes,
                result.update_ns_per_observation,
            )

        return tuple(
            candidate
            for candidate in self.results
            if not any(
                all(a <= b for a, b in zip(costs(other), costs(candidate), strict=True))
                and any(
                    a < b for a, b in zip(costs(other), costs(candidate), strict=True)
                )
                for other in self.results
            )
        )

    def select(
        self,
        *,
        objective: str = "brier",
        max_brier_degradation: float | None = None,
        max_serialized_bytes: int | None = None,
        max_update_ns_per_observation: float | None = None,
        max_predict_ms: float | None = None,
        max_update_ms: float | None = None,
    ) -> ConfigurationResult | None:
        """Minimize the chosen error subject to validation-measured budgets.

        Args:
            objective: ``brier``, ``log_loss``, or ``binned_calibration_error``.
            max_brier_degradation: Optional increase over the best Brier in this report.
            max_serialized_bytes: Optional nonnegative integer state budget in bytes.
            max_update_ns_per_observation: Optional measured update-time limit.
            max_predict_ms: Maximum p95 prediction-call latency in milliseconds.
            max_update_ms: Maximum p95 update-call latency in milliseconds.

        Returns:
            Lowest-error feasible result, breaking ties by size then name, or None.

        Raises:
            ValueError: If the objective is unknown or a budget is negative.
            TypeError: If the serialized-state budget is not an integer.
        """
        from streamcal._validation import nonnegative_finite

        if objective not in ("brier", "log_loss", "binned_calibration_error"):
            raise ValueError("unknown objective")
        margin = (
            np.inf
            if max_brier_degradation is None
            else nonnegative_finite(max_brier_degradation, name="max_brier_degradation")
        )
        for value, name in (
            (max_predict_ms, "max_predict_ms"),
            (max_update_ms, "max_update_ms"),
        ):
            if value is not None:
                nonnegative_finite(value, name=name)
        if max_serialized_bytes is not None:
            if isinstance(max_serialized_bytes, bool) or not isinstance(
                max_serialized_bytes, (int, np.integer)
            ):
                raise TypeError("max_serialized_bytes must be a nonnegative integer")
            if max_serialized_bytes < 0:
                raise ValueError("max_serialized_bytes must be a nonnegative integer")
        if max_update_ns_per_observation is not None:
            nonnegative_finite(
                max_update_ns_per_observation, name="max_update_ns_per_observation"
            )
        best = min(result.brier for result in self.results)
        feasible = [
            r
            for r in self.results
            if r.brier <= best + margin
            and (max_predict_ms is None or r.p95_predict_ms <= max_predict_ms)
            and (max_update_ms is None or r.p95_update_ms <= max_update_ms)
            and (
                max_serialized_bytes is None
                or r.serialized_bytes <= max_serialized_bytes
            )
            and (
                max_update_ns_per_observation is None
                or r.update_ns_per_observation <= max_update_ns_per_observation
            )
        ]
        return min(
            feasible,
            key=lambda r: (getattr(r, objective), r.serialized_bytes, r.name),
            default=None,
        )

    def paired_interval(
        self,
        candidate: str,
        reference: str,
        *,
        block_size: int,
        replicates: int = 2_000,
        seed: int = 2026,
    ) -> tuple[float, float, float]:
        """Return paired mean Brier difference and circular-block 95% interval.

        Requires the ``evaluation`` extra. Block size is in observations in
        prediction order. Choose it for the dependence scale of your stream.

        Args:
            candidate: Name of the candidate method.
            reference: Name of the reference method.
            block_size: Number of consecutive observations per bootstrap block.
            replicates: Number of bootstrap replicates.
            seed: Bootstrap random seed.

        Returns:
            Mean difference, lower endpoint, and upper endpoint.

        Raises:
            ValueError: If observations are absent or the block exceeds their count.
        """
        from arch.bootstrap import CircularBlockBootstrap

        positive_integer(block_size, name="block_size")
        positive_integer(replicates, name="replicates")
        by_name = {r.name: r for r in self.results}
        losses = np.asarray(by_name[candidate].squared_errors) - np.asarray(
            by_name[reference].squared_errors
        )
        if losses.size == 0 or block_size > losses.size:
            raise ValueError(
                "block_size must not exceed the nonempty observation count"
            )
        bootstrap = CircularBlockBootstrap(block_size, losses, seed=seed)
        interval = bootstrap.conf_int(np.mean, reps=replicates, method="percentile")
        return float(losses.mean()), float(interval[0, 0]), float(interval[1, 0])


def retained_array_bytes(calibrator: BaseCalibrator) -> int:
    """Return owned numerical-array storage across state and upstream models."""
    seen: set[int] = set()

    def visit(value: object) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return visit(value.base) if value.base is not None else value.nbytes
        if isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(v) for v in value)
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return visit(vars(value))
        return 0

    return visit(calibrator)


def compare_prequential(
    probabilities: ArrayLike,
    outcomes: ArrayLike,
    configurations: Mapping[
        str, StreamingIsotonicConfig | Callable[[], BaseCalibrator]
    ],
    *,
    batch_size: int,
    diagnostic_bins: int = 20,
    prediction_times: ArrayLike | None = None,
    label_delays: ArrayLike | None = None,
    observe_mask: ArrayLike | None = None,
) -> TradeoffReport:
    """Compare configurations in chronological predict-then-observe order.

    Timing fields describe this invocation on this machine. They are useful for
    local trade-off decisions, not portable performance guarantees.

    Args:
        probabilities: Raw probabilities in chronological order.
        outcomes: Binary outcomes aligned with ``probabilities``.
        configurations: Names mapped to streaming-isotonic configurations.
        batch_size: Number of observations predicted before each update.
        diagnostic_bins: Bin count for the descriptive calibration diagnostic.
        prediction_times: Nondecreasing prediction timestamps in seconds. When
            supplied, River schedules per-event prediction and label delivery;
            ``batch_size`` must be one. Requires the ``evaluation`` extra.
        label_delays: Nonnegative, finite delay per observation. Defaults to zero.
        observe_mask: Boolean vector controlling which labels may update a model.
            All forecasts are still scored offline; defaults to all True.

    Returns:
        Quality trajectories and resource measurements for every configuration.

    Raises:
        ValueError: If inputs, batch size, diagnostic bins, or configuration names
            are invalid.
    """
    probability_array, outcome_array = paired_binary_data(probabilities, outcomes)
    batch_size = positive_integer(batch_size, name="batch_size")
    diagnostic_bins = positive_integer(diagnostic_bins, name="diagnostic_bins")
    if not configurations:
        raise ValueError("configurations must not be empty")
    if any(not name for name in configurations):
        raise ValueError("configuration names must not be empty")
    mask = (
        np.ones(outcome_array.size, dtype=bool)
        if observe_mask is None
        else np.asarray(observe_mask)
    )
    if mask.shape != outcome_array.shape or mask.dtype != np.bool_:
        raise ValueError("observe_mask must be an aligned boolean vector")
    times = (
        None if prediction_times is None else np.asarray(prediction_times, dtype=float)
    )
    delays = (
        np.zeros(outcome_array.size)
        if label_delays is None
        else np.asarray(label_delays, dtype=float)
    )
    if times is None and label_delays is not None:
        raise ValueError("label_delays requires prediction_times")
    if times is not None:
        if (
            batch_size != 1
            or times.shape != outcome_array.shape
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) < 0)
        ):
            raise ValueError(
                "prediction_times must be finite, ordered and aligned; "
                "batch_size must be one"
            )
        if (
            delays.shape != times.shape
            or not np.all(np.isfinite(delays))
            or np.any(delays < 0)
            or not np.all(np.isfinite(times + delays))
        ):
            raise ValueError("label_delays must be finite, nonnegative and aligned")

    results: list[ConfigurationResult] = []
    for name, configuration in configurations.items():
        calibrator = (
            configuration.build()
            if isinstance(configuration, StreamingIsotonicConfig)
            else configuration()
        )
        if (
            isinstance(calibrator, StreamingIsotonicCalibrator)
            and calibrator.half_life_seconds is not None
            and times is None
        ):
            raise ValueError("half-life configurations require prediction_times")
        calibrated_batches: list[np.ndarray] = []
        batch_scores: list[float] = []
        cumulative_scores: list[float] = []
        cumulative_squared_error = 0.0
        cumulative_observations = 0
        predict_ns = 0
        update_ns = 0
        predict_calls: list[int] = []
        update_calls: list[int] = []

        for start in range(
            0, 0 if times is not None else probability_array.size, batch_size
        ):
            stop = min(start + batch_size, probability_array.size)
            batch_probabilities = probability_array[start:stop]
            batch_outcomes = outcome_array[start:stop]

            before = perf_counter_ns()
            calibrated = calibrator.calibrate(batch_probabilities)
            predict_calls.append(perf_counter_ns() - before)
            predict_ns += predict_calls[-1]
            calibrated_batches.append(calibrated)
            batch_score = brier_score(batch_outcomes, calibrated)
            batch_scores.append(batch_score)
            cumulative_squared_error += batch_score * calibrated.size
            cumulative_observations += calibrated.size
            cumulative_scores.append(cumulative_squared_error / cumulative_observations)

            before = perf_counter_ns()
            batch_mask = mask[start:stop]
            if np.any(batch_mask):
                calibrator.update(
                    batch_probabilities[batch_mask], batch_outcomes[batch_mask]
                )
            update_calls.append(perf_counter_ns() - before)
            update_ns += update_calls[-1]

        if times is not None:
            calibrated_all, predict_calls, update_calls = _run_delayed(
                calibrator, probability_array, outcome_array, times, delays, mask
            )
            predict_ns, update_ns = sum(predict_calls), sum(update_calls)
            batch_scores = [float(v) for v in (calibrated_all - outcome_array) ** 2]
            cumulative_scores = (
                np.cumsum(batch_scores) / np.arange(1, len(batch_scores) + 1)
            ).tolist()
        else:
            calibrated_all = np.concatenate(calibrated_batches)
        if np.unique(outcome_array).size == 2:
            auc = float(roc_auc_score(outcome_array, calibrated_all))
        else:
            auc = float("nan")
        results.append(
            ConfigurationResult(
                name=name,
                brier=brier_score(outcome_array, calibrated_all),
                log_loss=float(log_loss(outcome_array, calibrated_all, labels=[0, 1])),
                binned_calibration_error=binned_calibration_error(
                    outcome_array,
                    calibrated_all,
                    n_bins=diagnostic_bins,
                ),
                roc_auc=auc,
                predict_ns_per_observation=predict_ns / probability_array.size,
                update_ns_per_observation=update_ns / probability_array.size,
                state_bytes=retained_array_bytes(calibrator),
                batch_brier=tuple(batch_scores),
                cumulative_brier=tuple(cumulative_scores),
                serialized_bytes=len(pickle.dumps(calibrator, protocol=5)),
                squared_errors=tuple((calibrated_all - outcome_array) ** 2),
                p95_predict_ms=float(np.quantile(predict_calls, 0.95)) / 1e6,
                p95_update_ms=float(np.quantile(update_calls, 0.95)) / 1e6
                if update_calls
                else 0.0,
            )
        )

    return TradeoffReport(
        n_observations=probability_array.size,
        batch_size=batch_size,
        results=tuple(results),
    )


def _run_delayed(
    calibrator: BaseCalibrator,
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    times: np.ndarray,
    delays: np.ndarray,
    observe_mask: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    from river.stream import simulate_qa

    dataset: Dataset = (
        ({"p": p, "time": t, "delay": d}, y)
        for p, t, d, y in zip(probabilities, times, delays, outcomes, strict=True)
    )
    predictions = np.empty(outcomes.size)
    predict_calls: list[int] = []
    update_calls: list[int] = []
    timed = (
        isinstance(calibrator, StreamingIsotonicCalibrator)
        and calibrator.half_life_seconds is not None
    )
    for event in simulate_qa(dataset, moment="time", delay="delay"):
        index, features, label = event[:3]
        p = np.array([features["p"]])
        before = perf_counter_ns()
        if label is None:
            if timed and isinstance(calibrator, StreamingIsotonicCalibrator):
                predictions[index] = calibrator.calibrate(
                    p, current_time=features["time"]
                )[0]
            else:
                predictions[index] = calibrator.calibrate(p)[0]
            predict_calls.append(perf_counter_ns() - before)
        elif observe_mask[index]:
            if timed and isinstance(calibrator, StreamingIsotonicCalibrator):
                calibrator.update(
                    p,
                    [label],
                    prediction_times=[features["time"]],
                    observed_at=features["time"] + features["delay"],
                )
            else:
                calibrator.update(p, [label])
            update_calls.append(perf_counter_ns() - before)
    return predictions, predict_calls, update_calls
