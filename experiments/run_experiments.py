"""
Main experiment runner for MWU calibration experiments.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from methods import (
    MWUCalibrator,
    OnlineSGD,
    PerBucketEMA,
    NoCalibration,
    TemperatureScaling,
    IsotonicCalibrator,
    PlattScaling,
    create_calibrator,
)
from data_generators import (
    synthetic_stream,
    semi_synthetic_stream,
    production_scale_stream,
    DRIFT_SCENARIOS,
)
from metrics import (
    brier_score,
    expected_calibration_error,
    per_bucket_mse,
    compute_bucket_errors,
    time_to_recovery,
    compute_convergence_rate,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single_experiment(
    data_generator,
    calibrators: dict[str, Any],
    n_buckets: int = 100,
) -> dict[str, list]:
    """
    Run a single experiment with multiple calibrators.

    Args:
        data_generator: Iterator yielding (p_raw, y) batches
        calibrators: Dict mapping method names to calibrator instances
        n_buckets: Number of buckets for metrics

    Returns:
        Dict with per-batch metrics for each method
    """
    results = {name: [] for name in calibrators}
    bucket_errors_by_method = {name: [] for name in calibrators}

    for method_name, cal in calibrators.items():
        cal.reset()

    bins = np.linspace(0, 1, n_buckets + 1)

    for batch_idx, (p_raw, y) in enumerate(data_generator):
        for method_name, cal in calibrators.items():
            t0 = time.perf_counter()
            p_cal = cal.update(p_raw, y)
            cpu_time = time.perf_counter() - t0

            bucket_idx = np.clip(np.digitize(p_raw, bins) - 1, 0, n_buckets - 1)
            bucket_errors, bucket_counts = compute_bucket_errors(
                y, p_cal, bucket_idx, n_buckets
            )

            results[method_name].append({
                "batch": batch_idx,
                "brier": brier_score(y, p_cal),
                "ece": expected_calibration_error(y, p_cal),
                "bucket_mse": per_bucket_mse(bucket_errors, bucket_counts),
                "cpu_ms": cpu_time * 1000,
            })
            bucket_errors_by_method[method_name].append(bucket_errors)

    return results, bucket_errors_by_method


def run_synthetic_experiments(
    n_seeds: int = 20,
    n_total: int = 200_000,
    batch_size: int = 5_000,
    n_buckets: int = 100,
    eta: float = 0.1,
) -> dict:
    """Run experiments on synthetic data across all drift scenarios."""
    all_results = {}

    for drift_scenario in DRIFT_SCENARIOS.keys():
        print(f"  Running {drift_scenario} drift...")
        scenario_results = []

        for seed in range(n_seeds):
            data_gen = synthetic_stream(
                n_total=n_total,
                batch_size=batch_size,
                n_buckets=n_buckets,
                drift_scenario=drift_scenario,
                seed=seed,
            )

            calibrators = {
                "none": NoCalibration(),
                "ema": PerBucketEMA(n_buckets=n_buckets, alpha=0.3),
                "sgd": OnlineSGD(n_buckets=n_buckets, eta=eta),
                "mwu": MWUCalibrator(n_buckets=n_buckets, eta=eta),
                "temperature": TemperatureScaling(),
                "isotonic": IsotonicCalibrator(),
                "platt_1": PlattScaling(refit_every=1),
                "platt_5": PlattScaling(refit_every=5),
                "platt_10": PlattScaling(refit_every=10),
            }

            results, bucket_errors = run_single_experiment(
                data_gen, calibrators, n_buckets
            )
            scenario_results.append({"results": results, "bucket_errors": bucket_errors})

        all_results[drift_scenario] = scenario_results

    return all_results


def run_semi_synthetic_experiments(
    n_seeds: int = 20,
    n_train: int = 60_000,
    n_stream: int = 40_000,
    batch_size: int = 2_000,
    n_buckets: int = 50,
    eta: float = 0.1,
) -> dict:
    """Run experiments on semi-synthetic data."""
    all_results = {}

    for domain in ["ad_ctr", "income", "loan"]:
        print(f"  Running {domain} domain...")
        domain_results = []

        for seed in range(n_seeds):
            data_gen = semi_synthetic_stream(
                domain=domain,
                n_train=n_train,
                n_stream=n_stream,
                batch_size=batch_size,
                drift_scenario="linear_up",
                seed=seed,
            )

            calibrators = {
                "none": NoCalibration(),
                "ema": PerBucketEMA(n_buckets=n_buckets, alpha=0.3),
                "sgd": OnlineSGD(n_buckets=n_buckets, eta=eta),
                "mwu": MWUCalibrator(n_buckets=n_buckets, eta=eta),
                "temperature": TemperatureScaling(),
                "isotonic": IsotonicCalibrator(),
                "platt_1": PlattScaling(refit_every=1),
                "platt_5": PlattScaling(refit_every=5),
                "platt_10": PlattScaling(refit_every=10),
            }

            results, bucket_errors = run_single_experiment(
                data_gen, calibrators, n_buckets
            )
            domain_results.append({"results": results, "bucket_errors": bucket_errors})

        all_results[domain] = domain_results

    return all_results


def run_production_scale_experiment(
    n_seeds: int = 5,
    n_total: int = 1_000_000,
    batch_size: int = 10_000,
    n_buckets: int = 100,
    eta: float = 0.1,
) -> dict:
    """Run production-scale experiment for timing analysis."""
    all_results = []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...")
        data_gen = production_scale_stream(
            n_total=n_total,
            batch_size=batch_size,
            n_buckets=n_buckets,
            seed=seed,
        )

        calibrators = {
            "none": NoCalibration(),
            "ema": PerBucketEMA(n_buckets=n_buckets, alpha=0.3),
            "sgd": OnlineSGD(n_buckets=n_buckets, eta=eta),
            "mwu": MWUCalibrator(n_buckets=n_buckets, eta=eta),
            "temperature": TemperatureScaling(),
            "isotonic": IsotonicCalibrator(),
            "platt_1": PlattScaling(refit_every=1),
        }

        results, _ = run_single_experiment(data_gen, calibrators, n_buckets)
        all_results.append(results)

    return all_results


def run_convergence_experiment(
    n_seeds: int = 20,
    n_total: int = 200_000,
    batch_size: int = 5_000,
    n_buckets: int = 100,
) -> dict:
    """Run convergence experiment with diminishing step sizes."""
    all_results = []

    for seed in range(n_seeds):
        data_gen = synthetic_stream(
            n_total=n_total,
            batch_size=batch_size,
            drift_scenario="stationary",
            seed=seed,
        )

        calibrators = {
            "mwu_diminishing": MWUCalibrator(
                n_buckets=n_buckets, eta=0.5, diminishing_lr=True, mu=0.1
            ),
            "sgd_diminishing": OnlineSGD(n_buckets=n_buckets, eta=0.1),
            "mwu_constant": MWUCalibrator(n_buckets=n_buckets, eta=0.1),
        }

        results, bucket_errors = run_single_experiment(data_gen, calibrators, n_buckets)
        all_results.append({"results": results, "bucket_errors": bucket_errors})

    return all_results


def run_eta_sensitivity_experiment(
    n_seeds: int = 20,
    n_total: int = 200_000,
    batch_size: int = 5_000,
    n_buckets: int = 100,
) -> dict:
    """Run step size sensitivity experiment."""
    etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    all_results = {}

    for eta in etas:
        print(f"  eta = {eta}")
        eta_results = []

        for seed in range(n_seeds):
            data_gen = synthetic_stream(
                n_total=n_total,
                batch_size=batch_size,
                drift_scenario="linear_up",
                seed=seed,
            )

            calibrators = {
                "mwu": MWUCalibrator(n_buckets=n_buckets, eta=eta),
            }

            results, _ = run_single_experiment(data_gen, calibrators, n_buckets)
            eta_results.append(results)

        all_results[eta] = eta_results

    return all_results


def run_bucket_sensitivity_experiment(
    n_seeds: int = 20,
    n_total: int = 200_000,
    batch_size: int = 5_000,
) -> dict:
    """Run bucket count sensitivity experiment."""
    bucket_counts = [10, 25, 50, 100, 200]
    all_results = {}

    for n_buckets in bucket_counts:
        print(f"  B = {n_buckets}")
        bucket_results = []

        for seed in range(n_seeds):
            data_gen = synthetic_stream(
                n_total=n_total,
                batch_size=batch_size,
                drift_scenario="linear_up",
                seed=seed,
            )

            calibrators = {
                "mwu": MWUCalibrator(n_buckets=n_buckets, eta=0.1),
                "ema": PerBucketEMA(n_buckets=n_buckets, alpha=0.3),
                "isotonic": IsotonicCalibrator(),
            }

            results, _ = run_single_experiment(data_gen, calibrators, n_buckets)
            bucket_results.append(results)

        all_results[n_buckets] = bucket_results

    return all_results


def run_batch_size_sensitivity_experiment(
    n_seeds: int = 20,
    n_total: int = 200_000,
    n_buckets: int = 50,
) -> dict:
    """Run batch size sensitivity experiment."""
    batch_sizes = [200, 500, 1000, 2000, 5000]
    all_results = {}

    for batch_size in batch_sizes:
        print(f"  batch_size = {batch_size}")
        batch_results = []

        for seed in range(n_seeds):
            data_gen = synthetic_stream(
                n_total=n_total,
                batch_size=batch_size,
                drift_scenario="linear_up",
                seed=seed,
            )

            calibrators = {
                "mwu": MWUCalibrator(n_buckets=n_buckets, eta=0.1),
                "ema": PerBucketEMA(n_buckets=n_buckets, alpha=0.3),
                "isotonic": IsotonicCalibrator(),
            }

            results, _ = run_single_experiment(data_gen, calibrators, n_buckets)
            batch_results.append(results)

        all_results[batch_size] = batch_results

    return all_results


def aggregate_results(results_list: list[dict], method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate results across seeds."""
    values = []
    for seed_result in results_list:
        if isinstance(seed_result, dict) and "results" in seed_result:
            method_results = seed_result["results"][method]
        else:
            method_results = seed_result[method]
        metric_values = [r[metric] for r in method_results]
        values.append(metric_values)

    values = np.array(values)
    mean = np.mean(values, axis=0)
    ci = 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, ci


def get_final_metrics(results_list: list[dict], method: str) -> dict:
    """Get final batch metrics across seeds."""
    final_brier = []
    final_ece = []
    mean_cpu = []

    for seed_result in results_list:
        if isinstance(seed_result, dict) and "results" in seed_result:
            method_results = seed_result["results"][method]
        else:
            method_results = seed_result[method]

        final_brier.append(method_results[-1]["brier"])
        final_ece.append(method_results[-1]["ece"])
        mean_cpu.append(np.mean([r["cpu_ms"] for r in method_results]))

    return {
        "brier_mean": np.mean(final_brier),
        "brier_ci": 1.96 * np.std(final_brier) / np.sqrt(len(final_brier)),
        "ece_mean": np.mean(final_ece),
        "ece_ci": 1.96 * np.std(final_ece) / np.sqrt(len(final_ece)),
        "cpu_ms_mean": np.mean(mean_cpu),
        "cpu_ms_ci": 1.96 * np.std(mean_cpu) / np.sqrt(len(mean_cpu)),
    }


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(convert_numpy(results), f)
    print(f"Saved results to {filepath}")


def main():
    print("=" * 60)
    print("MWU Calibration Experiments")
    print("=" * 60)

    print("\n1. Running synthetic experiments (all drift scenarios)...")
    synthetic_results = run_synthetic_experiments()
    save_results(synthetic_results, "synthetic_results.json")

    print("\n2. Running semi-synthetic experiments...")
    semi_synthetic_results = run_semi_synthetic_experiments()
    save_results(semi_synthetic_results, "semi_synthetic_results.json")

    print("\n3. Running production-scale experiment...")
    production_results = run_production_scale_experiment()
    save_results(production_results, "production_results.json")

    print("\n4. Running convergence experiment (stationary, diminishing LR)...")
    convergence_results = run_convergence_experiment()
    save_results(convergence_results, "convergence_results.json")

    print("\n5. Running eta sensitivity experiment...")
    eta_results = run_eta_sensitivity_experiment()
    save_results(eta_results, "eta_sensitivity_results.json")

    print("\n6. Running bucket sensitivity experiment...")
    bucket_results = run_bucket_sensitivity_experiment()
    save_results(bucket_results, "bucket_sensitivity_results.json")

    print("\n7. Running batch size sensitivity experiment...")
    batch_size_results = run_batch_size_sensitivity_experiment()
    save_results(batch_size_results, "batch_size_sensitivity_results.json")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
