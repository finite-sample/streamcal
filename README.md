# streamcal

[![PyPI](https://img.shields.io/pypi/v/streamcal)](https://pypi.org/project/streamcal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Calibrate ML probability outputs in streaming settings.

## The Problem

Your model outputs 0.7 for a prediction. Is the true probability actually 70%? Usually not—most models are miscalibrated. You need to learn a correction function, but:

- Data arrives in batches (streaming)
- Distribution shifts over time
- You need monotonicity (if model says A > B, calibrated should too)

## Install

```bash
pip install streamcal
```

## Usage

```python
from streamcal import StreamingIsotonicCalibrator

cal = StreamingIsotonicCalibrator(n_buckets=100, alpha=0.1)

for batch in data_stream:
    p_calibrated = cal.update(batch.predictions, batch.outcomes)
    # Use p_calibrated for decisions
```

That's it. The calibrator:
- Tracks outcome rates per probability bucket via EMA
- Projects to monotonic function via PAV
- Adapts to drift (tune `alpha`: higher = faster adaptation)

## API

| Calibrator | Use Case |
|------------|----------|
| `StreamingIsotonicCalibrator` | Streaming, guaranteed monotonic |
| `NearlyIsotonicCalibrator` | Streaming, soft monotonicity |
| `IsotonicCalibrator` | Batch baseline |
| `PlattScaling` | Batch baseline |
| `TemperatureScaling` | Batch baseline |

```python
from streamcal import brier_score, expected_calibration_error

brier_score(y_true, y_pred)  # Lower is better
expected_calibration_error(y_true, y_pred)  # Lower is better
```

## License

MIT
