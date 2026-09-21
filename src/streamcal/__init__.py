"""Bounded-memory calibration for binary probability streams."""

from importlib.metadata import version

from streamcal.batch import BatchCalibrator
from streamcal.calibrators import CalibratorDiagnostics, StreamingIsotonicCalibrator
from streamcal.evaluation import (
    ConfigurationResult,
    StreamingIsotonicConfig,
    TradeoffReport,
    compare_prequential,
)
from streamcal.metrics import binned_calibration_error, brier_score

__version__ = version("streamcal")

__all__ = [
    "BatchCalibrator",
    "CalibratorDiagnostics",
    "ConfigurationResult",
    "StreamingIsotonicCalibrator",
    "StreamingIsotonicConfig",
    "TradeoffReport",
    "binned_calibration_error",
    "brier_score",
    "compare_prequential",
]
