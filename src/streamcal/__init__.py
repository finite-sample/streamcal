"""Bounded-memory calibration for binary probability streams."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from streamcal.calibrators import CalibratorDiagnostics, StreamingIsotonicCalibrator

if TYPE_CHECKING:
    from streamcal.batch import BatchCalibrator
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

# Names whose modules import scikit-learn. They load on first access, so a
# service that only runs StreamingIsotonicCalibrator never imports it.
_LAZY = {
    "BatchCalibrator": "streamcal.batch",
    "ConfigurationResult": "streamcal.evaluation",
    "StreamingIsotonicConfig": "streamcal.evaluation",
    "TradeoffReport": "streamcal.evaluation",
    "compare_prequential": "streamcal.evaluation",
    "binned_calibration_error": "streamcal.metrics",
    "brier_score": "streamcal.metrics",
}


def __getattr__(name: str) -> Any:
    """Import a public name from its module on first access.

    Args:
        name: Attribute requested from the package.

    Returns:
        The requested public object.

    Raises:
        AttributeError: If ``name`` is not a public name of the package.
    """
    if name not in _LAZY:
        raise AttributeError(f"module 'streamcal' has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public names, including those not yet imported."""
    return sorted(set(globals()) | set(__all__))
