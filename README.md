# MWU Calibration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Streaming probability calibration via multiplicative weights.

## Installation

```bash
pip install streamcal
```

For development:
```bash
pip install -e ".[dev]"
```

## The Problem

ML models output probabilities that are often miscalibrated—a predicted 70% doesn't mean 70% of those cases are positive. Batch calibrators (Platt scaling, isotonic regression) require periodic refits, creating a compute-drift tradeoff.

MWU maintains per-bucket bias factors with O(#buckets) cost per batch, adapting continuously without offline retraining.

## Method

Maintain bias factors $c_b$ per bucket. After each batch:

$$c_b \leftarrow c_b \cdot \exp(-\eta \cdot (\bar{p}_b - \bar{y}_b))$$

where $\bar{p}_b$ is the mean calibrated probability and $\bar{y}_b$ is the observed outcome rate in bucket $b$.

## Results

Semi-synthetic experiments (LightGBM base model, linear drift, B=50 buckets):

| Method | Brier | ECE | CPU ms/batch |
|--------|-------|-----|--------------|
| MWU | 0.133 | 0.070 | 0.08 |
| Platt | 0.129 | 0.043 | 4.92 |
| Isotonic | 0.128 | 0.043 | 4.36 |

MWU is **61× faster** than Platt while achieving comparable Brier scores.

## Usage

```python
from streamcal import MWUCalibrator

cal = MWUCalibrator(n_buckets=50, eta=0.1)

for p_raw, y in data_stream:
    p_calibrated = cal.update(p_raw, y)
```

### Available Calibrators

**Streaming (online):**
- `MWUCalibrator` - Multiplicative Weights Update
- `OnlineSGD` - Online SGD with additive updates
- `PerBucketEMA` - Per-bucket exponential moving average

**Batch (refit on accumulated data):**
- `PlattScaling` - Logistic regression on logits
- `IsotonicCalibrator` - Isotonic regression
- `TemperatureScaling` - Temperature scaling

### Metrics

```python
from streamcal import brier_score, expected_calibration_error

brier = brier_score(y_true, y_pred)
ece = expected_calibration_error(y_true, y_pred, n_bins=20)
```

## Reproduce Experiments

```bash
pip install -e ".[experiments]"
python experiments/run_experiments.py
python experiments/generate_figures.py
```

## Paper

See [ms/mwu_calibration.pdf](ms/mwu_calibration.pdf) for theory and full results.

## Related Work

This uses the same MWU/mirror descent algorithm as [onlinerake](https://github.com/soodoku/onlinerake) (survey weighting), applied to probability calibration instead of sample reweighting.

## License

MIT
