"""Faster code paths must reproduce the straightforward ones exactly."""

import subprocess
import sys

import numpy as np
import pytest

import streamcal
from streamcal import BatchCalibrator
from streamcal.calibrators import (
    _GRID_INDEX_FROM,
    _GRID_INTERP_FROM,
    _bin_index,
    _grid_index,
    _grid_interp,
    _interpolate,
)


def _adversarial_points(n_bins, rng):
    """Every edge and center, one float step either side of each, and noise."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    marks = np.concatenate([edges, centers])
    points = np.concatenate(
        [marks, np.nextafter(marks, 0.0), np.nextafter(marks, 1.0), rng.random(500)]
    )
    return edges, centers, np.clip(points, 0.0, 1.0)


@pytest.mark.parametrize("n_bins", [*range(1, 70), 100, 997, 1000, 4096])
def test_grid_kernels_match_numpy_bit_for_bit(n_bins):
    rng = np.random.default_rng(n_bins)
    edges, centers, points = _adversarial_points(n_bins, rng)
    reference = np.clip(np.digitize(points, edges) - 1, 0, n_bins - 1)
    assert np.array_equal(_grid_index(points, edges), reference)
    if n_bins >= 2:
        fitted = np.sort(rng.random(n_bins))
        fast = _grid_interp(points, centers, fitted)
        slow = np.interp(points, centers, fitted)
        assert np.array_equal(fast.view(np.int64), slow.view(np.int64))


@pytest.mark.parametrize("n_bins", [1, 2, 20, 1000])
def test_size_switches_agree_on_both_sides_of_the_threshold(n_bins):
    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fitted = np.sort(rng.random(n_bins))
    for size in (1, _GRID_INDEX_FROM - 1, _GRID_INDEX_FROM, _GRID_INTERP_FROM + 1):
        points = rng.random(size)
        assert np.array_equal(
            _bin_index(points, edges),
            np.clip(np.digitize(points, edges) - 1, 0, n_bins - 1),
        )
        assert np.array_equal(
            _interpolate(points, centers, fitted), np.interp(points, centers, fitted)
        )


@pytest.mark.parametrize("window", [None, 1, 7, 50])
def test_one_byte_labels_refit_on_exactly_the_retained_rows(window):
    rng = np.random.default_rng(1)
    chunked = BatchCalibrator(window_size=window)
    seen_p, seen_y = [], []
    for size in (40, 1, 13, 1, 60, 5, 30):
        p = rng.uniform(0.01, 0.99, size)
        y = (rng.random(size) < p).astype(float)
        y[-2:] = [0.0, 1.0][-size:]  # every chunk ends with both classes
        chunked.update(p, y)
        seen_p.append(p)
        seen_y.append(y)
    all_p, all_y = np.concatenate(seen_p), np.concatenate(seen_y)
    keep = all_p.size if window is None else window
    direct = BatchCalibrator().update(all_p[-keep:], all_y[-keep:])
    grid = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(chunked.calibrate(grid), direct.calibrate(grid))
    assert chunked.retained_observations == keep
    assert chunked.history_bytes == keep * (8 + 1)


def test_history_does_not_alias_caller_arrays():
    p = np.array([0.2, 0.4, 0.6, 0.8])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = BatchCalibrator(refit_every=2).update(p, y)
    p[:] = 0.5
    model.update([0.1, 0.9], [0, 1])
    direct = BatchCalibrator().update(
        [0.2, 0.4, 0.6, 0.8, 0.1, 0.9], [0, 0, 1, 1, 0, 1]
    )
    grid = np.linspace(0.0, 1.0, 11)
    assert np.array_equal(model.calibrate(grid), direct.calibrate(grid))


def test_importing_the_streaming_calibrator_does_not_load_scikit_learn():
    probe = (
        "import sys, streamcal\n"
        "streamcal.StreamingIsotonicCalibrator().update([0.2, 0.8], [0, 1])\n"
        "print('sklearn' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and probe
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_every_public_name_resolves():
    for name in streamcal.__all__:
        assert getattr(streamcal, name) is not None
    assert set(streamcal.__all__) <= set(dir(streamcal))
    with pytest.raises(AttributeError, match="no attribute"):
        _ = streamcal.not_a_name
