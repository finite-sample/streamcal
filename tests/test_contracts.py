"""Behavioral contracts for the public calibration API."""

from __future__ import annotations

import pickle
from functools import partial

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from streamcal import (
    BatchCalibrator,
    StreamingIsotonicCalibrator,
    binned_calibration_error,
    brier_score,
)
from streamcal.batch import ProbabilityClassifier

CALIBRATOR_FACTORIES = [
    StreamingIsotonicCalibrator,
    BatchCalibrator,
    partial(BatchCalibrator, "sigmoid"),
    partial(BatchCalibrator, "temperature"),
]


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_update_is_predict_then_observe(factory):
    """The current outcomes must not alter predictions already issued for them."""
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    before = factory().calibrate(probabilities)
    calibrator = factory()

    returned = calibrator.update(probabilities, np.array([0.0, 0.0, 1.0, 1.0]))

    assert returned is calibrator
    assert np.array_equal(before, probabilities)


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
@pytest.mark.parametrize(
    ("probabilities", "outcomes", "message"),
    [
        (np.array([]), np.array([]), "must not be empty"),
        (np.array([0.2, 0.8]), np.array([1.0]), "same length"),
        (np.array([-0.1, 0.8]), np.array([0.0, 1.0]), "between 0 and 1"),
        (np.array([0.2, 1.1]), np.array([0.0, 1.0]), "between 0 and 1"),
        (np.array([np.nan]), np.array([1.0]), "finite"),
        (np.array([0.5]), np.array([np.inf]), "finite"),
        (np.array([0.2, 0.8]), np.array([-1.0, 2.0]), "0 or 1"),
        (np.array([[0.2, 0.8]]), np.array([0.0, 1.0]), "one-dimensional"),
    ],
)
def test_update_rejects_invalid_data(factory, probabilities, outcomes, message):
    with pytest.raises(ValueError, match=message):
        factory().update(probabilities, outcomes)


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_calibrate_rejects_invalid_probabilities(factory):
    with pytest.raises(ValueError, match="between 0 and 1"):
        factory().calibrate(np.array([1.01]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bins": 0}, "n_bins"),
        ({"n_bins": True}, "n_bins"),
        ({"decay": -0.1}, "decay"),
        ({"decay": 1.1}, "decay"),
        ({"decay": np.nan}, "decay"),
        ({"prior_weight": -1.0}, "prior_weight"),
    ],
)
def test_streaming_parameters_are_validated(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        StreamingIsotonicCalibrator(**kwargs)


def test_streaming_no_forgetting_matches_weighted_bucket_reference():
    probabilities = np.array([0.05, 0.12, 0.18, 0.27, 0.43, 0.51, 0.74, 0.91])
    outcomes = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    calibrator = StreamingIsotonicCalibrator(n_bins=5, decay=1.0, prior_weight=0.0)
    calibrator.update(probabilities[:3], outcomes[:3])
    calibrator.update(probabilities[3:], outcomes[3:])

    indices = np.clip(np.digitize(probabilities, calibrator.bin_edges) - 1, 0, 4)
    counts = np.bincount(indices, minlength=5).astype(float)
    positives = np.bincount(indices, weights=outcomes, minlength=5)
    active = counts > 0
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        calibrator.bin_centers[active],
        positives[active] / counts[active],
        sample_weight=counts[active],
    )

    expected = reference.predict(calibrator.bin_centers)
    assert np.allclose(calibrator.calibration_values, expected)


def test_streaming_decay_matches_explicit_weighted_history():
    first_p = np.array([0.1, 0.3, 0.7, 0.9])
    first_y = np.array([0.0, 0.0, 1.0, 1.0])
    second_p = np.array([0.1, 0.3, 0.7, 0.9])
    second_y = np.array([1.0, 1.0, 0.0, 0.0])
    calibrator = StreamingIsotonicCalibrator(n_bins=4, decay=0.5, prior_weight=0.0)
    calibrator.update(first_p, first_y)
    calibrator.update(second_p, second_y)

    probabilities = np.concatenate([first_p, second_p])
    outcomes = np.concatenate([first_y, second_y])
    weights = np.concatenate([np.full(4, 0.5), np.ones(4)])
    indices = np.clip(np.digitize(probabilities, calibrator.bin_edges) - 1, 0, 3)
    counts = np.bincount(indices, weights=weights, minlength=4)
    positives = np.bincount(indices, weights=weights * outcomes, minlength=4)
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        calibrator.bin_centers,
        positives / counts,
        sample_weight=counts,
    )

    assert np.allclose(
        calibrator.calibration_values,
        reference.predict(calibrator.bin_centers),
    )


def test_streaming_state_size_is_independent_of_history_length():
    calibrator = StreamingIsotonicCalibrator(n_bins=20)
    rng = np.random.default_rng(7)
    initial_bytes = calibrator.diagnostics().state_bytes

    for _ in range(100):
        probabilities = rng.uniform(0.0, 1.0, 200)
        outcomes = rng.binomial(1, probabilities).astype(float)
        calibrator.update(probabilities, outcomes)

    diagnostics = calibrator.diagnostics()
    assert diagnostics.state_bytes == initial_bytes
    assert diagnostics.n_observations == 20_000
    assert diagnostics.n_updates == 100


def test_streaming_round_trip_preserves_predictions():
    calibrator = StreamingIsotonicCalibrator(n_bins=10)
    probabilities = np.linspace(0.05, 0.95, 20)
    outcomes = (probabilities > 0.5).astype(float)
    calibrator.update(probabilities, outcomes)

    restored = pickle.loads(pickle.dumps(calibrator))  # noqa: S301

    assert np.array_equal(
        restored.calibrate(probabilities), calibrator.calibrate(probabilities)
    )
    assert restored.diagnostics() == calibrator.diagnostics()


def test_batch_isotonic_matches_sklearn_reference():
    probabilities = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    outcomes = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    points = np.linspace(0.0, 1.0, 101)
    calibrator = BatchCalibrator().update(probabilities, outcomes)
    reference = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    ).fit(probabilities, outcomes)

    assert np.array_equal(calibrator.calibrate(points), reference.predict(points))


@pytest.mark.parametrize("method", ["sigmoid", "temperature"])
def test_parametric_matches_public_sklearn_reference(method):
    probabilities = np.array([0.05, 0.15, 0.35, 0.45, 0.65, 0.8, 0.9])
    outcomes = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    points = np.linspace(0.05, 0.95, 21)
    calibrator = BatchCalibrator(method).update(probabilities, outcomes)
    reference = CalibratedClassifierCV(
        FrozenEstimator(ProbabilityClassifier()),
        method=method,
        cv=[(np.arange(outcomes.size), np.arange(outcomes.size))],
        ensemble=False,
    ).fit(probabilities[:, None], outcomes)

    assert np.allclose(
        calibrator.calibrate(points),
        reference.predict_proba(points[:, None])[:, 1],
    )


def test_accumulating_references_copy_observed_arrays():
    probabilities = np.array([0.2, 0.8])
    outcomes = np.array([0.0, 1.0])
    calibrator = BatchCalibrator().update(probabilities, outcomes)
    original = calibrator.calibrate(np.array([0.2, 0.8]))

    probabilities[:] = 0.5
    outcomes[:] = 1.0

    assert np.array_equal(
        calibrator.calibrate(np.array([0.2, 0.8])),
        original,
    )


@pytest.mark.parametrize(
    "metric",
    [brier_score, binned_calibration_error],
)
def test_metrics_reject_broadcastable_length_mismatch(metric):
    with pytest.raises(ValueError, match="same length"):
        metric(np.array([1.0]), np.array([0.2, 0.8]))


def test_binned_calibration_error_rejects_nonpositive_bin_count():
    with pytest.raises(ValueError, match="n_bins"):
        binned_calibration_error(np.array([0.0, 1.0]), np.array([0.2, 0.8]), n_bins=0)


def test_brier_score_matches_sklearn_reference():
    outcomes = np.array([0.0, 1.0, 1.0, 0.0])
    probabilities = np.array([0.1, 0.6, 0.9, 0.4])

    assert brier_score(outcomes, probabilities) == brier_score_loss(
        outcomes,
        probabilities,
    )


@pytest.mark.parametrize("refit_every", [0, -1, 1.5, True])
def test_platt_refit_cadence_is_validated(refit_every):
    with pytest.raises((TypeError, ValueError), match="refit_every"):
        BatchCalibrator("sigmoid", refit_every=refit_every)
