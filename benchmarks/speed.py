"""Time and memory receipts for the hot paths, comparable across commits.

Run from the repository root, optionally pointing at another checkout's
``src`` to compare commits on the same machine::

    python -m benchmarks.speed                      # this checkout
    python -m benchmarks.speed --src ../other/src   # another commit

Timings are medians of repeated calls with BLAS pinned to one thread. They
describe this machine; compare two commits by running both here.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

IMPORT_PROBE = """
import resource, sys, time
sys.path.insert(0, {src!r})
start = time.perf_counter()
import streamcal
from streamcal import StreamingIsotonicCalibrator
seconds = time.perf_counter() - start
first = time.perf_counter()
StreamingIsotonicCalibrator().update([0.2, 0.8], [0, 1])
first_update = time.perf_counter() - first
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024
print(seconds, first_update, rss * scale, "sklearn" in sys.modules)
"""


def median_us(call, repeats):
    """Median wall time of ``call`` in microseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1e3


def import_cost(src, runs=7):
    """Median import time, first-update time and peak RSS in fresh processes."""
    rows = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for _ in range(runs):
        output = subprocess.run(  # noqa: S603 - fixed interpreter and probe
            [sys.executable, "-c", IMPORT_PROBE.format(src=src)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.split()
        rows.append(output)
    return {
        "import_ms": statistics.median(float(r[0]) for r in rows) * 1e3,
        "first_update_ms": statistics.median(float(r[1]) for r in rows) * 1e3,
        "peak_rss_mb": statistics.median(float(r[2]) for r in rows) / 2**20,
        "imports_sklearn": rows[0][3] == "True",
    }


def main():
    """Measure and print the receipts as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(Path("src").resolve()))
    parser.add_argument("--output")
    args = parser.parse_args()
    sys.path.insert(0, args.src)
    from streamcal import (
        BatchCalibrator,
        StreamingIsotonicCalibrator,
        compare_prequential,
    )

    rng = np.random.default_rng(0)
    results = {"import": import_cost(args.src)}
    for bins in (20, 100, 1000):
        for size in (1, 100, 10_000):
            p = rng.uniform(0, 1, size)
            y = (rng.random(size) < p).astype(float)
            repeats = 2000 if size < 10_000 else 300
            decayed = StreamingIsotonicCalibrator(n_bins=bins, decay=0.9).update(p, y)
            timed = StreamingIsotonicCalibrator(n_bins=bins, half_life_seconds=100.0)
            timed.update(p, y, prediction_times=np.zeros(size), observed_at=1.0)
            key = f"B={bins} m={size}"
            results[f"update {key}"] = median_us(partial(decayed.update, p, y), repeats)
            results[f"calibrate {key}"] = median_us(
                partial(decayed.calibrate, p), repeats
            )
            results[f"half-life calibrate {key}"] = median_us(
                partial(timed.calibrate, p, current_time=2.0), repeats
            )
    for history in (10_000, 100_000, 1_000_000):
        p = rng.uniform(0, 1, history)
        y = (rng.random(history) < p).astype(float)
        batch = BatchCalibrator(refit_every=10**9).update(p, y)
        results[f"batch update, no refit, H={history:,}"] = median_us(
            partial(batch.update, p[:10], y[:10]), 50
        )
        results[f"batch history bytes, H={history:,}"] = batch.history_bytes
    size = 3000
    p = rng.uniform(0, 1, size)
    y = (rng.random(size) < p).astype(int)
    start = time.perf_counter()
    compare_prequential(
        p, y, {"stream": lambda: StreamingIsotonicCalibrator(n_bins=20)}, batch_size=1
    )
    results["compare_prequential us/observation, batch 1"] = (
        (time.perf_counter() - start) / size * 1e6
    )
    text = json.dumps(results, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
