"""Time and memory receipts for the hot paths, comparable across commits.

Run from the repository root, optionally pointing at another checkout's
``src`` to compare commits on the same machine::

    python -m benchmarks.speed                      # this checkout
    python -m benchmarks.speed --src ../other/src   # another commit

Timings are medians of repeated calls with BLAS pinned to one thread. They
describe this machine; compare two commits by running both here.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path


def median_us(call, repeats):
    """Median wall time of ``call`` in microseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1e3


def probe_import(src):
    """Time ``import streamcal`` and its first update in this (fresh) process.

    Import time only means something in a process that has not imported
    anything yet, so :func:`import_cost` runs this in a new interpreter each
    time, through ``--probe-import``. That is also why this module imports
    NumPy inside :func:`main` rather than at the top.
    """
    import resource

    sys.path.insert(0, src)
    start = time.perf_counter()
    from streamcal import StreamingIsotonicCalibrator

    imported = time.perf_counter()
    StreamingIsotonicCalibrator().update([0.2, 0.8], [0, 1])
    updated = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "import_ms": (imported - start) * 1e3,
        "first_update_ms": (updated - imported) * 1e3,
        # ru_maxrss is bytes on macOS and kilobytes on Linux.
        "peak_rss_mb": peak * (1 if sys.platform == "darwin" else 1024) / 2**20,
        "imports_sklearn": "sklearn" in sys.modules,
    }


def import_cost(src, runs=7):
    """Median of :func:`probe_import` over fresh interpreters."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "benchmarks.speed", "--probe-import"]
    rows = [
        json.loads(
            subprocess.run(  # noqa: S603 - this script, run by the same interpreter
                [*command, "--src", src],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            ).stdout
        )
        for _ in range(runs)
    ]
    return {
        key: statistics.median(row[key] for row in rows)
        for key in ("import_ms", "first_update_ms", "peak_rss_mb")
    } | {"imports_sklearn": rows[0]["imports_sklearn"]}


def main():
    """Measure and print the receipts as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(Path("src").resolve()))
    parser.add_argument("--output")
    parser.add_argument("--probe-import", action="store_true", help="internal")
    args = parser.parse_args()
    if args.probe_import:
        print(json.dumps(probe_import(args.src)))
        return
    # Probe imports before this process loads anything heavy: on Linux a child's
    # peak-memory counter starts from the parent's size when it is forked.
    results = {"import": import_cost(args.src)}
    import numpy as np

    sys.path.insert(0, args.src)
    from streamcal import (
        BatchCalibrator,
        StreamingIsotonicCalibrator,
        compare_prequential,
    )

    rng = np.random.default_rng(0)
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
