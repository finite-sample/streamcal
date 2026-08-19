"""streamcal: Streaming probability calibration with monotonicity guarantee."""

from importlib.metadata import version

from streamcal.batch import IsotonicCalibrator, PlattScaling, TemperatureScaling
from streamcal.calibrators import NearlyIsotonicCalibrator, StreamingIsotonicCalibrator
from streamcal.metrics import brier_score, expected_calibration_error

__version__ = version("streamcal")

__all__ = [
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "PlattScaling",
    "StreamingIsotonicCalibrator",
    "TemperatureScaling",
    "brier_score",
    "expected_calibration_error",
]
