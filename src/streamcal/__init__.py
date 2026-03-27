"""
streamcal: Streaming probability calibration with monotonicity guarantee.
"""

from streamcal.batch import IsotonicCalibrator, PlattScaling, TemperatureScaling
from streamcal.calibrators import NearlyIsotonicCalibrator, StreamingIsotonicCalibrator
from streamcal.metrics import brier_score, expected_calibration_error

__version__ = "0.2.0"

__all__ = [
    "StreamingIsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "IsotonicCalibrator",
    "PlattScaling",
    "TemperatureScaling",
    "brier_score",
    "expected_calibration_error",
]
