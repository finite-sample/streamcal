"""Tests for calibrators."""

import numpy as np
import pytest

from streamcal import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    PlattScaling,
    StreamingIsotonicCalibrator,
    TemperatureScaling,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    p_raw = np.random.uniform(0.2, 0.8, 100)
    y = (np.random.random(100) < p_raw).astype(float)
    return p_raw, y


class TestStreamingIsotonicCalibrator:
    def test_init(self):
        cal = StreamingIsotonicCalibrator(n_buckets=50)
        assert cal.n_buckets == 50
        assert len(cal.calibration_values) == 50

    def test_init_with_alpha(self):
        cal = StreamingIsotonicCalibrator(n_buckets=50, alpha=0.2)
        assert cal.alpha == 0.2

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = StreamingIsotonicCalibrator(n_buckets=10)
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_calibrate_uses_learned_params(self, sample_data):
        p_raw, y = sample_data
        cal = StreamingIsotonicCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        p_cal = cal.calibrate(p_raw)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = StreamingIsotonicCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        cal.reset()
        assert np.allclose(cal.ema_rates, cal.bucket_centers)
        assert not np.any(cal.has_seen)

    def test_monotonicity_after_multiple_updates(self):
        """Verify calibration output is always monotonic."""
        cal = StreamingIsotonicCalibrator(n_buckets=50)

        rng = np.random.default_rng(42)
        for _ in range(100):
            p_raw = rng.uniform(0, 1, 100)
            y = (rng.random(100) < p_raw * 0.8 + 0.1).astype(float)
            cal.update(p_raw, y)

        test_p = np.linspace(0, 1, 1000)
        calibrated = cal.calibrate(test_p)
        assert np.all(np.diff(calibrated) >= -1e-10), "Calibration must be monotonic"

    def test_monotonicity_with_adversarial_data(self):
        """Test monotonicity even with non-monotonic raw outcome rates."""
        cal = StreamingIsotonicCalibrator(n_buckets=10)

        p_raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        for _ in range(10):
            cal.update(p_raw, y)

        test_p = np.linspace(0, 1, 100)
        calibrated = cal.calibrate(test_p)
        assert np.all(np.diff(calibrated) >= -1e-10), "Must enforce monotonicity"

    def test_rank_preservation(self, sample_data):
        """Verify that if p1 > p2, then calibrate(p1) >= calibrate(p2)."""
        p_raw, y = sample_data
        cal = StreamingIsotonicCalibrator(n_buckets=20)

        for _ in range(50):
            cal.update(p_raw, y)

        p_test = np.array([0.3, 0.5, 0.7])
        p_cal = cal.calibrate(p_test)
        assert p_cal[0] <= p_cal[1] <= p_cal[2], "Rank must be preserved"

    def test_drift_adaptation(self):
        """Test that calibrator adapts to distributional drift."""
        cal = StreamingIsotonicCalibrator(n_buckets=20, alpha=0.3)
        rng = np.random.default_rng(42)

        for _ in range(50):
            p_raw = rng.uniform(0.4, 0.6, 200)
            y = (rng.random(200) < 0.3).astype(float)
            cal.update(p_raw, y)

        mid_calibration = cal.calibrate(np.array([0.5]))[0]

        for _ in range(50):
            p_raw = rng.uniform(0.4, 0.6, 200)
            y = (rng.random(200) < 0.8).astype(float)
            cal.update(p_raw, y)

        new_calibration = cal.calibrate(np.array([0.5]))[0]

        assert new_calibration > mid_calibration, "Should adapt to drift"


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


class TestNearlyIsotonicCalibrator:
    def test_init(self):
        cal = NearlyIsotonicCalibrator(n_buckets=50, lam=1.0, alpha=0.1)
        assert cal.n_buckets == 50
        assert cal.lam == 1.0
        assert cal.alpha == 0.1
        assert len(cal.calibration_values) == 50

    def test_update_returns_probabilities(self, sample_data):
        p_raw, y = sample_data
        cal = NearlyIsotonicCalibrator(n_buckets=10)
        p_cal = cal.update(p_raw, y)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_calibrate_uses_learned_params(self, sample_data):
        p_raw, y = sample_data
        cal = NearlyIsotonicCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        p_cal = cal.calibrate(p_raw)
        assert len(p_cal) == len(p_raw)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_reset_clears_state(self, sample_data):
        p_raw, y = sample_data
        cal = NearlyIsotonicCalibrator(n_buckets=10)
        cal.update(p_raw, y)
        original_centers = cal.bucket_centers.copy()
        cal.reset()
        assert np.allclose(cal.calibration_values, original_centers)
        assert np.allclose(cal.ema_rates, original_centers)
        assert not np.any(cal.has_seen)

    def test_high_lambda_encourages_monotonicity(self):
        """High lambda should push towards monotonic solution."""
        cal = NearlyIsotonicCalibrator(n_buckets=10, lam=10.0, alpha=0.1)

        rng = np.random.default_rng(42)
        for _ in range(200):
            p_raw = rng.uniform(0, 1, 200)
            y = (rng.random(200) < p_raw).astype(float)
            cal.update(p_raw, y)

        diffs = np.diff(cal.calibration_values)
        violations = np.sum(diffs < -1e-6)
        assert violations <= 5, f"High lambda: few violations, got {violations}"

    def test_drift_adaptation(self):
        """Test that calibrator adapts to distributional drift."""
        cal = NearlyIsotonicCalibrator(n_buckets=20, lam=1.0, alpha=0.3)
        rng = np.random.default_rng(42)

        for _ in range(100):
            p_raw = rng.uniform(0.4, 0.6, 200)
            y = (rng.random(200) < 0.3).astype(float)
            cal.update(p_raw, y)

        mid_calibration = cal.calibrate(np.array([0.5]))[0]

        for _ in range(100):
            p_raw = rng.uniform(0.4, 0.6, 200)
            y = (rng.random(200) < 0.8).astype(float)
            cal.update(p_raw, y)

        new_calibration = cal.calibrate(np.array([0.5]))[0]

        assert new_calibration > mid_calibration, "Should adapt to drift"
