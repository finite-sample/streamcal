"""Tests for metrics."""

import numpy as np

from streamcal import brier_score, expected_calibration_error


class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_pred) == 0.0

    def test_worst_predictions(self):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_pred) == 1.0

    def test_intermediate_predictions(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        assert brier_score(y_true, y_pred) == 0.25


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self):
        np.random.seed(42)
        n = 10000
        y_pred = np.random.uniform(0, 1, n)
        y_true = (np.random.random(n) < y_pred).astype(float)
        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        assert ece < 0.05

    def test_miscalibrated(self):
        y_true = np.ones(100)
        y_pred = np.full(100, 0.3)
        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        assert ece > 0.5

    def test_returns_float(self):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.3, 0.7, 0.3, 0.7])
        ece = expected_calibration_error(y_true, y_pred)
        assert isinstance(ece, float)
