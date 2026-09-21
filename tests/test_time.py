"""Exact weighted-history checks for prediction-age forgetting."""

import pickle

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from streamcal import BatchCalibrator, StreamingIsotonicCalibrator


def test_time_weights_match_explicit_history_with_late_labels():
    model = StreamingIsotonicCalibrator(n_bins=4, half_life_seconds=10, prior_weight=0)
    model.update([0.1, 0.9], [0, 1], prediction_times=[0, 10], observed_at=10)
    model.update([0.1, 0.9], [1, 0], prediction_times=[5, 20], observed_at=20)
    weights = np.exp2(-(20 - np.array([0, 10, 5, 20])) / 10)
    reference = IsotonicRegression(out_of_bounds="clip").fit(
        [0.125, 0.875, 0.125, 0.875], [0, 1, 1, 0], sample_weight=weights
    )
    assert np.allclose(model.calibration_values, reference.predict(model.bin_centers))
    assert model.diagnostics().effective_observations == pytest.approx(weights.sum())


def test_single_bin_has_closed_form_prior_weighted_expected_result():
    model = StreamingIsotonicCalibrator(n_bins=1, half_life_seconds=10, prior_weight=2)
    model.update([0.1, 0.9], [1, 0], prediction_times=[0, 10], observed_at=10)
    # Label weights are 1/2 and 1; prior contributes mass 2 and positive mass 1.
    expected = (0.5 + 1.0) / (0.5 + 1.0 + 2.0)
    assert model.calibrate([0.1, 0.9], current_time=10) == pytest.approx(
        [expected, expected]
    )


def test_time_partition_equivalence_and_prediction_immutability():
    together = StreamingIsotonicCalibrator(n_bins=4, half_life_seconds=10)
    separate = StreamingIsotonicCalibrator(n_bins=4, half_life_seconds=10)
    together.update([0.1, 0.9], [1, 0], prediction_times=[0, 5], observed_at=10)
    separate.update([0.1], [1], prediction_times=[0], observed_at=10)
    separate.update([0.9], [0], prediction_times=[5], observed_at=10)
    assert np.array_equal(together.calibration_values, separate.calibration_values)
    before = pickle.dumps(together)
    now = together.calibrate([0.125], current_time=10)[0]
    future = together.calibrate([0.125], current_time=100)[0]
    assert abs(future - 0.125) < abs(now - 0.125)
    assert pickle.dumps(together) == before
    restored = pickle.loads(before)  # noqa: S301
    assert restored.calibrate([0.125], current_time=100)[0] == future
    together.reset()
    assert together.last_observed_at is None
    assert together.calibrate([0.2], current_time=-5)[0] == 0.2


@pytest.mark.parametrize("half_life", [0, -1, np.nan, np.inf, True])
def test_invalid_half_life(half_life):
    with pytest.raises((ValueError, TypeError)):
        StreamingIsotonicCalibrator(half_life_seconds=half_life)


def test_time_validation_is_atomic():
    with pytest.raises(ValueError, match="mutually exclusive"):
        StreamingIsotonicCalibrator(decay=0.9, half_life_seconds=10)
    model = StreamingIsotonicCalibrator(half_life_seconds=10)
    model.update([0.5], [1], prediction_times=[1], observed_at=2)
    before = pickle.dumps(model)
    for times, arrival in [
        ([3], 2),
        ([0], 1),
        ([np.nan], 3),
        ([], 3),
        ([0], np.inf),
        ([0], None),
    ]:
        with pytest.raises(ValueError, match=r"prediction_times|timestamp|observed_at"):
            model.update([0.5], [1], prediction_times=times, observed_at=arrival)
        assert pickle.dumps(model) == before
    with pytest.raises(ValueError, match="timestamp"):
        model.calibrate([0.5])
    with pytest.raises(ValueError, match="timestamps"):
        StreamingIsotonicCalibrator().update([0.5], [1], observed_at=1)


def test_rolling_reference_matches_only_retained_window():
    p = np.linspace(0.05, 0.95, 20)
    y = np.tile([0, 1], 10)
    rolling = (
        BatchCalibrator(window_size=8).update(p[:10], y[:10]).update(p[10:], y[10:])
    )
    direct = BatchCalibrator().update(p[-8:], y[-8:])
    assert np.array_equal(rolling.calibrate(p), direct.calibrate(p))
    assert rolling.retained_observations == 8
    assert rolling.history_bytes == 8 * 16
    assert rolling.n_observations == 20


@pytest.mark.parametrize(
    "kwargs", [{"method": "bad"}, {"window_size": 0}, {"window_size": True}]
)
def test_batch_policy_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BatchCalibrator(**kwargs)


def test_zero_weight_history_uses_the_declared_center_extension():
    model = StreamingIsotonicCalibrator(n_bins=4, half_life_seconds=1, prior_weight=0)
    model.update([0.2], [1], prediction_times=[0], observed_at=2000)
    assert np.array_equal(model.calibration_values, model.bin_centers)
    with pytest.raises(ValueError, match="requires"):
        StreamingIsotonicCalibrator().update([0.2], [1], prediction_times=[0])
