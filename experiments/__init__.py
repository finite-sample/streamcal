"""
MWU Calibration Experiments

This package contains the experimental code for the paper
"Always-On Probability Calibration With Multiplicative Weights"

Modules:
    methods: Calibration methods (MWU, Platt, Isotonic, EMA, etc.)
    data_generators: Data generation (synthetic, semi-synthetic, production)
    metrics: Evaluation metrics (Brier, ECE, per-bucket MSE)
    run_experiments: Main experiment runner
    generate_figures: Figure and table generation
"""

from .methods import (
    MWUCalibrator,
    OnlineSGD,
    PerBucketEMA,
    NoCalibration,
    TemperatureScaling,
    IsotonicCalibrator,
    PlattScaling,
    create_calibrator,
)

from .data_generators import (
    synthetic_stream,
    semi_synthetic_stream,
    production_scale_stream,
    DRIFT_SCENARIOS,
)

from .metrics import (
    brier_score,
    expected_calibration_error,
    per_bucket_mse,
    compute_bucket_errors,
    time_to_recovery,
    compute_convergence_rate,
)

__all__ = [
    "MWUCalibrator",
    "OnlineSGD",
    "PerBucketEMA",
    "NoCalibration",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "PlattScaling",
    "create_calibrator",
    "synthetic_stream",
    "semi_synthetic_stream",
    "production_scale_stream",
    "DRIFT_SCENARIOS",
    "brier_score",
    "expected_calibration_error",
    "per_bucket_mse",
    "compute_bucket_errors",
    "time_to_recovery",
    "compute_convergence_rate",
]
