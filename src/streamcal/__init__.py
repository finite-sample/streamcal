"""
streamcal: Streaming probability calibration via multiplicative weights.
"""

from streamcal.batch import IsotonicCalibrator, PlattScaling, TemperatureScaling
from streamcal.calibrators import MWUCalibrator, OnlineSGD, PerBucketEMA
from streamcal.metrics import brier_score, expected_calibration_error

__version__ = "0.1.0"

__all__ = [
    "MWUCalibrator",
    "OnlineSGD",
    "PerBucketEMA",
    "PlattScaling",
    "IsotonicCalibrator",
    "TemperatureScaling",
    "brier_score",
    "expected_calibration_error",
]
