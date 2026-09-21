"""Tests for calibration diagnostics."""

import numpy as np

from streamcal import binned_calibration_error, brier_score


def test_brier_score_known_values():
    outcomes = np.array([0.0, 1.0, 0.0, 1.0])

    assert brier_score(outcomes, outcomes) == 0.0
    assert brier_score(outcomes, 1.0 - outcomes) == 1.0
    assert brier_score(outcomes, np.full(4, 0.5)) == 0.25


def test_binned_calibration_error_known_value():
    outcomes = np.ones(100)
    probabilities = np.full(100, 0.3)

    assert np.isclose(binned_calibration_error(outcomes, probabilities), 0.7)


def test_binned_calibration_error_is_a_float():
    outcomes = np.array([0.0, 1.0, 0.0, 1.0])
    probabilities = np.array([0.3, 0.7, 0.3, 0.7])

    assert isinstance(binned_calibration_error(outcomes, probabilities), float)
