"""Isolated-process measurements of state, peak RSS and call latency."""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import psutil

from benchmarks.policies import OnlineLogisticReference
from benchmarks.suite import environment
from streamcal import BatchCalibrator, StreamingIsotonicCalibrator
from streamcal.evaluation import retained_array_bytes


def measure(method, n, batch_size):
    """Measure one isolated process, generating input one batch at a time."""
    factories = {
        "stream": lambda: StreamingIsotonicCalibrator(n_bins=20),
        "rolling": lambda: BatchCalibrator(window_size=336, refit_every=16),
        "accumulating": lambda: BatchCalibrator(refit_every=16),
        "online-logistic": OnlineLogisticReference,
    }
    model = factories[method]()
    rng = np.random.default_rng(2026)
    baseline = psutil.Process().memory_info().rss
    prediction_ns, update_ns = [], []
    for start in range(0, n, batch_size):
        p = rng.uniform(0.05, 0.95, min(batch_size, n - start))
        y = rng.binomial(1, p**2)
        before = perf_counter_ns()
        model.calibrate(p)
        prediction_ns.append(perf_counter_ns() - before)
        before = perf_counter_ns()
        model.update(p, y)
        update_ns.append(perf_counter_ns() - before)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return {
        "method": method,
        "observations": n,
        "batch_size": batch_size,
        "numeric_array_bytes": retained_array_bytes(model),
        "serialized_bytes": len(pickle.dumps(model, protocol=5)),
        "baseline_rss_bytes": baseline,
        "peak_process_rss_bytes": peak,
        "mean_predict_ns_per_observation": sum(prediction_ns) / n,
        "mean_update_ns_per_observation": sum(update_ns) / n,
        "p95_predict_call_ms": float(np.quantile(prediction_ns, 0.95) / 1e6),
        "p95_update_call_ms": float(np.quantile(update_ns, 0.95) / 1e6),
    }


def main():
    """Run each benchmark in a fresh interpreter, without concurrent timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        method, n, batch = args.worker
        print(json.dumps(measure(method, int(n), int(batch))))
        return
    if args.output is None:
        parser.error("--output is required outside worker mode")
    records = []
    for n, batch in [(n, batch) for n in (1000, 10000) for batch in (1, 100, 1000)] + [
        (100000, 1000)
    ]:
        for method in ("stream", "rolling", "accumulating", "online-logistic"):
            for repeat in range(2):
                run = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.resources",
                        "--worker",
                        method,
                        str(n),
                        str(batch),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                records.append({"repeat": repeat, **json.loads(run.stdout)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"environment": environment(), "measurements": records}, indent=2)
        + "\n"
    )
    print(f"Resource evidence written to {args.output}")


if __name__ == "__main__":
    main()
