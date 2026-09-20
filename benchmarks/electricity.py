"""Chronological Elec2 evidence benchmark.

Run with ``uv run --group evidence python -m benchmarks.electricity``. The data
are pinned by OpenML identifier, version, URL, and checksum. Results are printed
as JSON so they can be archived with the exact commit and machine metadata.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import numpy as np
import pooch
import sklearn
from arch.bootstrap import CircularBlockBootstrap
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from streamcal import (
    BatchCalibrator,
    StreamingIsotonicConfig,
    binned_calibration_error,
    brier_score,
    compare_prequential,
)
from streamcal.evaluation import retained_array_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from streamcal.calibrators import BaseCalibrator

DATASET_ID = 151
DATASET_VERSION = 1
DATASET_URL = "https://openml.org/data/v1/download/2419/electricity.arff"
DATASET_MD5 = "8ca97867d960ae029ae3a9ac2c923d34"
BATCH_SIZE = 336
RANDOM_SEED = 2026


@dataclass(frozen=True)
class MethodResult:
    """Quality and resource evidence for one method."""

    brier: float
    log_loss: float
    binned_calibration_error: float
    roc_auc: float
    median_predict_ns_per_observation: float
    median_update_ns_per_observation: float
    p95_update_ns_per_observation: float
    state_bytes: int


@dataclass(frozen=True)
class RunResult:
    """Predictions and timing from a chronological benchmark pass."""

    predictions: NDArray[np.float64]
    predict_ns_per_observation: NDArray[np.float64]
    update_ns_per_observation: NDArray[np.float64]


def _dataset_path() -> Path:
    path = pooch.retrieve(
        url=DATASET_URL,
        known_hash=f"md5:{DATASET_MD5}",
        fname="electricity-151-v1.arff",
        path=pooch.os_cache("streamcal"),
        progressbar=False,
    )
    return Path(path)


def _load_data(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    records, _ = arff.loadarff(path)
    feature_names = records.dtype.names[:-1]
    features = np.column_stack(
        [np.asarray(records[name], dtype=np.float64) for name in feature_names]
    )
    outcomes = np.asarray(records["class"] == b"UP", dtype=np.float64)
    return features, outcomes


def _warm(
    calibrator: BaseCalibrator,
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> None:
    for start in range(0, probabilities.size, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, probabilities.size)
        calibrator.calibrate(probabilities[start:stop])
        calibrator.update(probabilities[start:stop], outcomes[start:stop])


def _run(
    calibrator: BaseCalibrator,
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> RunResult:
    prediction_batches: list[NDArray[np.float64]] = []
    predict_times: list[float] = []
    update_times: list[float] = []
    for start in range(0, probabilities.size, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, probabilities.size)
        batch_probabilities = probabilities[start:stop]
        batch_outcomes = outcomes[start:stop]

        before = perf_counter_ns()
        calibrated = calibrator.calibrate(batch_probabilities)
        predict_times.append((perf_counter_ns() - before) / calibrated.size)
        prediction_batches.append(calibrated)

        before = perf_counter_ns()
        calibrator.update(batch_probabilities, batch_outcomes)
        update_times.append((perf_counter_ns() - before) / calibrated.size)

    return RunResult(
        predictions=np.concatenate(prediction_batches),
        predict_ns_per_observation=np.asarray(predict_times),
        update_ns_per_observation=np.asarray(update_times),
    )


def _score(
    outcomes: NDArray[np.float64],
    run: RunResult,
    state_bytes: int,
) -> MethodResult:
    return MethodResult(
        brier=brier_score(outcomes, run.predictions),
        log_loss=float(log_loss(outcomes, run.predictions)),
        binned_calibration_error=binned_calibration_error(
            outcomes,
            run.predictions,
            n_bins=20,
        ),
        roc_auc=float(roc_auc_score(outcomes, run.predictions)),
        median_predict_ns_per_observation=float(
            np.median(run.predict_ns_per_observation)
        ),
        median_update_ns_per_observation=float(
            np.median(run.update_ns_per_observation)
        ),
        p95_update_ns_per_observation=float(
            np.quantile(run.update_ns_per_observation, 0.95)
        ),
        state_bytes=state_bytes,
    )


def _moving_block_interval(
    outcomes: NDArray[np.float64],
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    block_batches: int = 4,
    replicates: int = 2_000,
) -> dict[str, float]:
    losses = (candidate - outcomes) ** 2 - (reference - outcomes) ** 2
    batch_sums = np.asarray(
        [
            losses[start : start + BATCH_SIZE].sum()
            for start in range(0, losses.size, BATCH_SIZE)
        ]
    )
    counts = np.asarray(
        [
            len(losses[start : start + BATCH_SIZE])
            for start in range(0, losses.size, BATCH_SIZE)
        ]
    )
    bootstrap = CircularBlockBootstrap(
        block_batches, batch_sums, counts, seed=RANDOM_SEED
    )
    interval = bootstrap.conf_int(
        lambda sums, sizes: sums.sum() / sizes.sum(),
        reps=replicates,
        method="percentile",
    )
    lower, upper = interval[:, 0]
    return {
        "mean_brier_difference": float(losses.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def _learning_curve(
    outcomes: NDArray[np.float64],
    predictions: Mapping[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    n_batches = int(np.ceil(outcomes.size / BATCH_SIZE))
    checkpoints = sorted({1, 4, 13, 26, 52, n_batches})
    curve = []
    for batches in checkpoints:
        stop = min(batches * BATCH_SIZE, outcomes.size)
        curve.append(
            {
                "observed_labels": stop,
                "brier": {
                    name: brier_score(outcomes[:stop], values[:stop])
                    for name, values in predictions.items()
                },
            }
        )
    return curve


def run_evidence(path: Path | None = None) -> dict[str, Any]:
    """Run the pinned chronological evidence study and return serializable data."""
    features, outcomes = _load_data(path or _dataset_path())
    train_stop = int(0.2 * outcomes.size)
    validation_stop = int(0.3 * outcomes.size)
    model = LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED)
    model.fit(features[:train_stop], outcomes[:train_stop])
    raw_probabilities = model.predict_proba(features)[:, 1]

    configurations = {
        f"bins={n_bins},decay={decay},prior={prior_weight}": (
            StreamingIsotonicConfig(
                n_bins=n_bins,
                decay=decay,
                prior_weight=prior_weight,
            )
        )
        for n_bins in (20, 50, 100)
        for decay in (0.5, 0.9, 0.99, 1.0)
        for prior_weight in (1.0, 10.0, 100.0)
    }
    validation = compare_prequential(
        raw_probabilities[train_stop:validation_stop],
        outcomes[train_stop:validation_stop],
        configurations,
        batch_size=BATCH_SIZE,
    )
    selected = min(validation.results, key=lambda result: result.brier)
    selected_config = configurations[selected.name]

    factories: dict[str, Callable[[], BaseCalibrator]] = {
        "streamcal": selected_config.build,
        "batch_isotonic": BatchCalibrator,
        "sigmoid": lambda: BatchCalibrator("sigmoid"),
        "temperature": lambda: BatchCalibrator("temperature"),
    }
    validation_probabilities = raw_probabilities[train_stop:validation_stop]
    validation_outcomes = outcomes[train_stop:validation_stop]
    test_probabilities = raw_probabilities[validation_stop:]
    test_outcomes = outcomes[validation_stop:]

    runs: dict[str, RunResult] = {}
    scores: dict[str, MethodResult] = {}
    for name, factory in factories.items():
        calibrator = factory()
        _warm(calibrator, validation_probabilities, validation_outcomes)
        run = _run(calibrator, test_probabilities, test_outcomes)
        runs[name] = run
        state_bytes = retained_array_bytes(calibrator)
        scores[name] = _score(test_outcomes, run, state_bytes)

    raw_run = RunResult(
        predictions=test_probabilities,
        predict_ns_per_observation=np.zeros(1),
        update_ns_per_observation=np.zeros(1),
    )
    scores["raw"] = _score(test_outcomes, raw_run, 0)
    predictions = {"raw": test_probabilities}
    predictions.update({name: run.predictions for name, run in runs.items()})
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "openml_id": DATASET_ID,
            "version": DATASET_VERSION,
            "md5": DATASET_MD5,
            "n_observations": int(outcomes.size),
            "train_stop": train_stop,
            "validation_stop": validation_stop,
            "batch_size": BATCH_SIZE,
        },
        "selected_configuration": {
            "name": selected.name,
            "validation_brier": selected.brier,
            **asdict(selected_config),
        },
        "test": {name: asdict(score) for name, score in scores.items()},
        "paired_brier_intervals": {
            "streamcal_minus_raw": _moving_block_interval(
                test_outcomes,
                runs["streamcal"].predictions,
                test_probabilities,
            ),
            "streamcal_minus_batch_isotonic": _moving_block_interval(
                test_outcomes,
                runs["streamcal"].predictions,
                runs["batch_isotonic"].predictions,
            ),
        },
        "learning_curve": _learning_curve(test_outcomes, predictions),
    }


def validate_claims(result: Mapping[str, Any]) -> None:
    """Validate result integrity without requiring a benchmark winner."""
    test = result["test"]
    for name, score in test.items():
        if not np.isfinite(score["brier"]) or not 0 <= score["brier"] <= 1:
            raise AssertionError(f"invalid Brier for {name}")


def main() -> None:
    """Print the complete evidence result as deterministic, sorted JSON."""
    result = run_evidence()
    validate_claims(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
