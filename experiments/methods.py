"""
Calibration methods for experiments.

This module imports calibrators from the streamcal package and adds
experiment-specific utilities.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from streamcal import (
    IsotonicCalibrator,
    MWUCalibrator,
    OnlineSGD,
    PerBucketEMA,
    PlattScaling,
    TemperatureScaling,
)
from streamcal.calibrators import BaseCalibrator

__all__ = [
    "BaseCalibrator",
    "NoCalibration",
    "PerBucketEMA",
    "OnlineSGD",
    "MWUCalibrator",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "PlattScaling",
    "create_calibrator",
]


class NoCalibration(BaseCalibrator):
    """No calibration - returns raw probabilities unchanged."""

    def calibrate(
        self, p_raw: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        return p_raw.copy()

    def update(
        self, p_raw: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        return p_raw.copy()

    def reset(self) -> None:
        pass


def create_calibrator(method: str, **kwargs: float | int | bool) -> BaseCalibrator:
    """Factory function for calibrators."""
    calibrators: dict[str, type[BaseCalibrator]] = {
        "none": NoCalibration,
        "ema": PerBucketEMA,
        "sgd": OnlineSGD,
        "mwu": MWUCalibrator,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
        "platt": PlattScaling,
    }

    if method not in calibrators:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(calibrators.keys())}"
        )

    return calibrators[method](**kwargs)
