"""Tests for calibrators."""

import numpy as np
import pytest

from streamcal import (
    IsotonicCalibrator,
    MWUCalibrator,
    OnlineSGD,
    PerBucketEMA,
    PlattScaling,
    TemperatureScaling,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    p_raw = np.random.uniform(0.2, 0.8, 100)
    y = (np.random.random(100) < p_raw).astype(float)
    return p_raw, y


class TestMWUCalibrator:
    def test_init(self):
        cal = MWUCalibrator(n_buckets=50, eta=0.1)
        assert cal.n_buckets == 50
        assert cal.eta == 0.1
        assert len(cal.weights) == 50

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = MWUCalibrator(n_buckets=10)
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_calibrate_uses_learned_params(self, sample_data):
        p_raw, y = sample_data
        cal = MWUCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        p_cal = cal.calibrate(p_raw)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = MWUCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        cal.reset()
        assert np.allclose(cal.weights, 1.0)
        assert cal.t == 0

    def test_diminishing_lr(self, sample_data):
        p_raw, y = sample_data
        cal = MWUCalibrator(n_buckets=10, diminishing_lr=True, mu=0.1)
        p_cal1 = cal.update(p_raw, y)
        p_cal2 = cal.update(p_raw, y)
        assert p_cal1 is not None
        assert p_cal2 is not None


class TestOnlineSGD:
    def test_init(self):
        cal = OnlineSGD(n_buckets=50, eta=0.1)
        assert cal.n_buckets == 50
        assert cal.eta == 0.1

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = OnlineSGD(n_buckets=10)
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = OnlineSGD(n_buckets=10)
        cal.update(p_raw, y)
        cal.reset()
        assert np.allclose(cal.theta, 0.0)


class TestPerBucketEMA:
    def test_init(self):
        cal = PerBucketEMA(n_buckets=50, alpha=0.3)
        assert cal.n_buckets == 50
        assert cal.alpha == 0.3

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = PerBucketEMA(n_buckets=10)
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = PerBucketEMA(n_buckets=10)
        cal.update(p_raw, y)
        cal.reset()
        assert np.allclose(cal.ema_rates, 0.5)
        assert not np.any(cal.has_seen)


class TestTemperatureScaling:
    def test_init(self):
        cal = TemperatureScaling()
        assert cal.temperature == 1.0

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = TemperatureScaling()
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = TemperatureScaling()
        cal.update(p_raw, y)
        cal.reset()
        assert cal.temperature == 1.0
        assert len(cal.all_logits) == 0


class TestIsotonicCalibrator:
    def test_init(self):
        cal = IsotonicCalibrator()
        assert not cal.fitted

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = IsotonicCalibrator()
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = IsotonicCalibrator()
        cal.update(p_raw, y)
        cal.reset()
        assert not cal.fitted


class TestPlattScaling:
    def test_init(self):
        cal = PlattScaling(refit_every=5)
        assert cal.refit_every == 5
        assert not cal.fitted

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = PlattScaling()
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = PlattScaling()
        cal.update(p_raw, y)
        cal.reset()
        assert not cal.fitted
        assert cal.batch_count == 0
