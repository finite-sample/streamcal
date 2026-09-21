"""Focused behavior tests for public calibrators."""

from functools import partial

import numpy as np
import pytest

from streamcal import (
    BatchCalibrator,
    StreamingIsotonicCalibrator,
)


@pytest.fixture
def sample_data():
    """Return a deterministic, nondegenerate binary calibration sample."""
    rng = np.random.default_rng(42)
    probabilities = rng.uniform(0.2, 0.8, 100)
    outcomes = rng.binomial(1, probabilities).astype(float)
    return probabilities, outcomes


def test_streaming_starts_as_exact_identity():
    calibrator = StreamingIsotonicCalibrator(n_bins=10)
    probabilities = np.array([0.0, 0.15, 0.5, 0.85, 1.0])

    assert np.array_equal(calibrator.calibrate(probabilities), probabilities)


def test_streaming_output_is_bounded_and_monotonic(sample_data):
    probabilities, outcomes = sample_data
    calibrator = StreamingIsotonicCalibrator(n_bins=20)
    calibrator.update(probabilities, outcomes)

    calibrated = calibrator.calibrate(np.linspace(0.0, 1.0, 1_000))

    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_streaming_reset_restores_initial_state(sample_data):
    probabilities, outcomes = sample_data
    calibrator = StreamingIsotonicCalibrator(n_bins=10)
    calibrator.update(probabilities, outcomes)

    returned = calibrator.reset()

    assert returned is calibrator
    assert np.array_equal(calibrator.calibrate(probabilities), probabilities)
    assert not calibrator.diagnostics().is_ready


def test_lower_decay_adapts_more_to_a_new_regime():
    probabilities = np.linspace(0.4, 0.6, 500)
    old_outcomes = np.zeros(500)
    new_outcomes = np.ones(500)
    stable = StreamingIsotonicCalibrator(n_bins=10, decay=0.99)
    adaptive = StreamingIsotonicCalibrator(n_bins=10, decay=0.1)
    for calibrator in (stable, adaptive):
        for _ in range(20):
            calibrator.update(probabilities, old_outcomes)
        calibrator.update(probabilities, new_outcomes)

    point = np.array([0.5])
    assert adaptive.calibrate(point)[0] > stable.calibrate(point)[0]


@pytest.mark.parametrize(
    "factory",
    [
        BatchCalibrator,
        partial(BatchCalibrator, "sigmoid"),
        partial(BatchCalibrator, "temperature"),
    ],
)
def test_batch_references_fit_then_reset(factory, sample_data):
    probabilities, outcomes = sample_data
    calibrator = factory()
    calibrator.update(probabilities, outcomes)

    calibrated = calibrator.calibrate(probabilities)

    assert calibrator.is_ready
    assert calibrated.shape == probabilities.shape
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert calibrator.reset() is calibrator
    assert not calibrator.is_ready
    assert np.array_equal(calibrator.calibrate(probabilities), probabilities)


@pytest.mark.parametrize(
    "factory",
    [partial(BatchCalibrator, "sigmoid"), partial(BatchCalibrator, "temperature")],
)
def test_parametric_references_wait_for_both_classes(factory):
    probabilities = np.array([0.1, 0.2, 0.3])
    calibrator = factory()

    calibrator.update(probabilities, np.zeros(3))

    assert not calibrator.is_ready
    assert np.array_equal(calibrator.calibrate(probabilities), probabilities)


def test_platt_refit_cadence_is_respected(sample_data):
    probabilities, outcomes = sample_data
    calibrator = BatchCalibrator("sigmoid", refit_every=2)

    calibrator.update(probabilities, outcomes)
    assert not calibrator.is_ready
    calibrator.update(probabilities, outcomes)
    assert calibrator.is_ready


def test_temperature_improves_log_loss_on_temperature_distortion():
    rng = np.random.default_rng(11)
    true_probabilities = rng.uniform(0.05, 0.95, 20_000)
    outcomes = rng.binomial(1, true_probabilities).astype(float)
    logits = np.log(true_probabilities / (1.0 - true_probabilities))
    distorted = 1.0 / (1.0 + np.exp(-0.5 * logits))
    calibrator = BatchCalibrator("temperature").update(distorted, outcomes)

    calibrated = calibrator.calibrate(distorted)
    epsilon = np.finfo(float).eps

    def log_loss(values):
        safe = np.clip(values, epsilon, 1.0 - epsilon)
        return -np.mean(outcomes * np.log(safe) + (1.0 - outcomes) * np.log1p(-safe))

    assert log_loss(calibrated) < log_loss(distorted)
    legacy_grid = np.logspace(-1.0, 1.0, 50)
    legacy_losses = []
    distorted_logits = np.log(distorted) - np.log1p(-distorted)
    for temperature in legacy_grid:
        scaled = distorted_logits / temperature
        legacy_losses.append(
            float(np.mean(np.logaddexp(0.0, scaled) - outcomes * scaled))
        )
    assert log_loss(calibrated) < min(legacy_losses) - 1e-6
