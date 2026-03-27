# MWU Calibration

Streaming probability calibration via multiplicative weights.

## The Problem

Batch calibrators (Platt scaling, isotonic regression) require periodic refits, creating a compute-drift tradeoff. MWU updates per-bucket bias factors with O(#buckets) cost per batch, adapting continuously without offline jobs.

## Method

Maintain bias factors $c_b$ per bucket. After each batch:

$$c_b \leftarrow c_b \cdot \exp(-\eta \cdot (\tilde{r}_b - \hat{r}_b))$$

where $\tilde{r}_b$ is the mean calibrated probability and $\hat{r}_b$ is the observed outcome rate.

## Key Results

From semi-synthetic experiments (LightGBM base model, linear drift, B=50):

| Method | Brier | ECE | CPU ms/batch |
|--------|-------|-----|--------------|
| MWU | 0.133 | 0.070 | 0.08 |
| Platt (every batch) | 0.129 | 0.043 | 4.92 |
| Isotonic | 0.128 | 0.043 | 4.36 |

MWU is **61× faster** than Platt and **54× faster** than isotonic while achieving comparable Brier scores.

## Quick Start

```python
from experiments.methods import MWUCalibrator

cal = MWUCalibrator(n_buckets=50, eta=0.1)

for p_raw, y in data_stream:
    p_calibrated = cal.update(p_raw, y)
```

## Reproduce

```bash
python experiments/run_experiments.py
python experiments/generate_figures.py
```

## Paper

See [ms/mwu_calibration.pdf](ms/mwu_calibration.pdf) for theory and full experimental results.
