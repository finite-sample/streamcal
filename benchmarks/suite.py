"""Validation-selected evidence against rolling, frozen and online references."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmarks.electricity import DATASET_MD5, _dataset_path, _load_data
from benchmarks.policies import OnlineLogisticReference, RawReference
from streamcal import BatchCalibrator, StreamingIsotonicCalibrator, compare_prequential

SEED = 2026


def environment():
    """Identify the runtime and exact local implementation used for the run."""
    digest = hashlib.sha256()
    paths = sorted(
        [
            *Path("src").rglob("*.py"),
            *Path("benchmarks").glob("*.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        ]
    )
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": digest.hexdigest(),
        **{
            package: version(package)
            for package in ("numpy", "scikit-learn", "river", "arch", "streamcal")
        },
    }


def candidates():
    """Return fixed search grids chosen before the final evaluation period."""
    factories = {"raw": RawReference}
    for bins in (20, 50, 100):
        for decay in (0.5, 0.9, 0.99, 1.0):
            for prior in (1.0, 10.0, 100.0):
                factories[f"stream/b={bins}/d={decay}/p={prior}"] = partial(
                    StreamingIsotonicCalibrator,
                    n_bins=bins,
                    decay=decay,
                    prior_weight=prior,
                )
    for method in ("isotonic", "sigmoid", "temperature"):
        factories[f"accumulating-{method}"] = partial(BatchCalibrator, method)
        factories[f"frozen-{method}"] = partial(BatchCalibrator, method)
        for window in (336, 1344, 5376):
            for cadence in (1, 4, 16):
                factories[f"rolling-{method}/w={window}/r={cadence}"] = partial(
                    BatchCalibrator, method, window_size=window, refit_every=cadence
                )
    for rate in (0.001, 0.01, 0.1):
        factories[f"online-logistic/lr={rate}"] = partial(OnlineLogisticReference, rate)
    return factories


def warmed(factory, name, probabilities, outcomes, batch_size):
    """Construct fresh state using only a prefix preceding scored predictions."""
    model = factory()
    for start in range(0, len(outcomes), batch_size):
        model.update(
            probabilities[start : start + batch_size],
            outcomes[start : start + batch_size],
        )
    if name.startswith("frozen-"):
        model.freeze()
    return model


def summary(result):
    """Return scalar measurements without duplicating observation-level arrays."""
    return {
        key: value
        for key, value in asdict(result).items()
        if key not in ("squared_errors", "batch_brier", "cumulative_brier")
    }


def choose_families(report):
    """Choose within each predeclared family using validation Brier only."""
    selected = {}
    for result in report.results:
        family = result.name.split("/")[0]
        if family not in selected or result.brier < selected[family].brier:
            selected[family] = result
    return {family: result.name for family, result in selected.items()}


def electricity():
    """Evaluate frozen validation choices on the final chronological period."""
    features, outcomes = _load_data(_dataset_path())
    train_stop, validation_stop = int(0.2 * len(outcomes)), int(0.3 * len(outcomes))
    warm_stop = (train_stop + validation_stop) // 2
    base = LogisticRegression(max_iter=2000, random_state=SEED).fit(
        features[:train_stop], outcomes[:train_stop]
    )
    probabilities = base.predict_proba(features)[:, 1]
    factories = candidates()
    validation_factories = {
        name: partial(
            warmed,
            factory,
            name,
            probabilities[train_stop:warm_stop],
            outcomes[train_stop:warm_stop],
            336,
        )
        for name, factory in factories.items()
    }
    validation = compare_prequential(
        probabilities[warm_stop:validation_stop],
        outcomes[warm_stop:validation_stop],
        validation_factories,
        batch_size=336,
    )
    selected = choose_families(validation)
    operating_points = []
    for objective in ("brier", "binned_calibration_error"):
        for budget_ms, budget_bytes in ((0.1, 2048), (1.0, 65536), (100.0, 65536)):
            choice = validation.select(
                objective=objective,
                max_update_ms=budget_ms,
                max_predict_ms=100,
                max_serialized_bytes=budget_bytes,
            )
            operating_points.append(
                {
                    "objective": objective,
                    "max_update_ms": budget_ms,
                    "max_predict_ms": 100,
                    "max_serialized_bytes": budget_bytes,
                    "selected": None if choice is None else choice.name,
                }
            )
    selected_names = set(selected.values()) | {
        p["selected"] for p in operating_points if p["selected"] is not None
    }
    test_factories = {
        name: partial(
            warmed,
            factories[name],
            name,
            probabilities[train_stop:validation_stop],
            outcomes[train_stop:validation_stop],
            336,
        )
        for name in sorted(selected_names)
    }
    test = compare_prequential(
        probabilities[validation_stop:],
        outcomes[validation_stop:],
        test_factories,
        batch_size=336,
    )
    stream_name = selected["stream"]
    intervals = {
        r.name: test.paired_interval(
            stream_name, r.name, block_size=1344, replicates=2000, seed=SEED
        )
        for r in test.results
        if r.name in selected.values() and r.name != stream_name
    }
    by_name = {r.name: r for r in test.results}
    for point in operating_points:
        result = by_name.get(point["selected"])
        point["test"] = None if result is None else summary(result)
        point["test_resource_feasible"] = (
            None
            if result is None
            else (
                result.p95_update_ms <= point["max_update_ms"]
                and result.p95_predict_ms <= point["max_predict_ms"]
                and result.serialized_bytes <= point["max_serialized_bytes"]
            )
        )
    checkpoints = (336, 1344, 4368, 8736, 17472, len(outcomes) - validation_stop)
    return {
        "dataset": {"openml_id": 151, "version": 1, "md5": DATASET_MD5},
        "split": {
            "train_stop": train_stop,
            "warm_stop": warm_stop,
            "validation_stop": validation_stop,
        },
        "batch_size": 336,
        "selected": selected,
        "validation": [summary(r) for r in validation.results],
        "test": [summary(r) for r in test.results],
        "paired_brier_intervals": intervals,
        "operating_points": operating_points,
        "learning_curve": [
            {
                "test_labels": n,
                "brier": {
                    r.name: float(np.mean(r.squared_errors[:n])) for r in test.results
                },
            }
            for n in checkpoints
        ],
    }


def controlled():
    """Measure known-truth behavior and delayed/sparse labels over fixed seeds."""
    records = []
    for seed in (17, 41, 93):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.05, 0.95, 2000)
        for scenario in (
            "calibrated",
            "stationary",
            "abrupt",
            "gradual",
            "sparse",
            "delayed",
        ):
            truth = p.copy() if scenario == "calibrated" else p**2
            if scenario in ("abrupt", "gradual", "delayed"):
                mix = (
                    (np.arange(p.size) >= 1000).astype(float)
                    if scenario != "gradual"
                    else np.linspace(0, 1, p.size)
                )
                truth = (1 - mix) * p**2 + mix * np.sqrt(p)
            y = rng.binomial(1, truth).astype(float)
            factories = {
                "raw": RawReference,
                "stream": partial(
                    StreamingIsotonicCalibrator,
                    n_bins=20,
                    half_life_seconds=200,
                    prior_weight=1,
                ),
                "rolling-isotonic": partial(
                    BatchCalibrator, window_size=336, refit_every=32
                ),
                "online-logistic": OnlineLogisticReference,
            }
            mask = (
                np.arange(p.size) % 10 == 0
                if scenario == "sparse"
                else np.ones(p.size, dtype=bool)
            )
            delays = (
                rng.integers(0, 400, p.size)
                if scenario == "delayed"
                else np.zeros(p.size)
            )
            report = compare_prequential(
                p,
                y,
                factories,
                batch_size=1,
                prediction_times=np.arange(p.size),
                label_delays=delays,
                observe_mask=mask,
            )
            for result in report.results:
                # Binary labels determine prediction from squared error uniquely.
                pred = np.where(
                    y == 0,
                    np.sqrt(result.squared_errors),
                    1 - np.sqrt(result.squared_errors),
                )
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        **summary(result),
                        "known_truth_mse": float(np.mean((pred - truth) ** 2)),
                    }
                )
    return records


def main():
    """Write the reproducible evidence artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "environment": environment(),
        "electricity": electricity(),
        "controlled": controlled(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Evidence written to {args.output}")


if __name__ == "__main__":
    main()
