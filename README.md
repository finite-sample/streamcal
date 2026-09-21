# streamcal

[![CI](https://github.com/finite-sample/streamcal/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/streamcal/actions/workflows/ci.yml)
[![Docs](https://github.com/finite-sample/streamcal/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/streamcal/)
[![PyPI](https://img.shields.io/pypi/v/streamcal)](https://pypi.org/project/streamcal/)
[![Python](https://img.shields.io/pypi/pyversions/streamcal)](https://pypi.org/project/streamcal/)
[![License: MIT](https://img.shields.io/pypi/l/streamcal)](https://github.com/finite-sample/streamcal/blob/main/LICENSE)

Keep a deployed model's binary probabilities calibrated as outcomes arrive,
without retaining its prediction history.

`streamcal` learns a monotone correction from outcomes as they arrive. It keeps
fixed-size per-bin sufficient statistics, can forget stale history, and makes
the prediction-before-observation order explicit so a forecast never uses its
own outcome.

The numerical fit is scikit-learn's weighted isotonic regression. Streamcal adds
the compact streaming state, prediction-age forgetting, and evaluation needed
to choose a quality/resource operating point.

## Install

```bash
pip install streamcal
```

## Predict, then observe

```python
from streamcal import StreamingIsotonicCalibrator

calibrator = StreamingIsotonicCalibrator(
    n_bins=50,
    decay=0.9,
    prior_weight=1.0,
)

for raw_probabilities, outcomes in stream:
    calibrated = calibrator.calibrate(raw_probabilities)
    make_decisions(calibrated)

    # Call only after these outcomes become available.
    calibrator.update(raw_probabilities, outcomes)
```

`calibrate` only sees earlier outcomes. `update` changes future predictions and
returns the calibrator, never in-sample fitted predictions.

## Choose the trade-off on your stream

The useful settings are visible rather than hidden:

- `n_bins` trades local flexibility for data per bin and memory.
- `decay` controls adaptation versus stability. `1.0` retains all history;
  smaller values respond more strongly to the newest batch.
- `prior_weight` shrinks sparse bins toward the identity map during cold start.

Use the prequential evaluator to compare settings in chronological order:

```python
from streamcal import StreamingIsotonicConfig, compare_prequential

report = compare_prequential(
    raw_probabilities,
    outcomes,
    {
        "stable": StreamingIsotonicConfig(n_bins=50, decay=0.99),
        "balanced": StreamingIsotonicConfig(n_bins=50, decay=0.9),
        "adaptive": StreamingIsotonicConfig(n_bins=50, decay=0.5),
    },
    batch_size=500,
)

for result in report.results:
    print(result.name, result.brier, result.log_loss, result.state_bytes)

choice = report.select(
    objective="binned_calibration_error",
    max_predict_ms=100,
    max_update_ms=1,
    max_serialized_bytes=65_536,
)
if choice is not None:
    print(choice.name, choice.brier, choice.p95_predict_ms)
```

The report includes per-batch and cumulative Brier scores, log loss, a binned
calibration diagnostic, AUROC, local timing, and state size. Timing values
describe the machine running the comparison; they are not portable promises.
The budgets above constrain the measured **p95 whole-call latency**, for the
batch size passed to the evaluator. Selection minimizes the requested error
among feasible configurations; it returns `None` if no choice fits the budgets.
Use `objective="brier"` or `"log_loss"` for a proper score. Binned error can favor
uninformative constant forecasts, so inspect the proper scores alongside it.

Select on a validation period, freeze the choice, and evaluate on a later
period. `max_brier_degradation` optionally limits the increase over the best
validation Brier. The report also exposes a Pareto frontier and paired Brier
intervals. Factories for `BatchCalibrator` or another `BaseCalibrator` can be
included in the same comparison. This offline evaluator retains predictions
and losses; its memory is separate from the calibrator's bounded state.

## Time-based forgetting and delayed labels

```python
from streamcal import StreamingIsotonicCalibrator

calibrator = StreamingIsotonicCalibrator(n_bins=50, half_life_seconds=86_400)
issued = calibrator.calibrate([0.3, 0.8], current_time=1_000)

# The caller stores the original probabilities and prediction timestamps.
calibrator.update(
    [0.3, 0.8],
    [0, 1],
    prediction_times=[1_000, 1_000],
    observed_at=2_000,
)
```

An old prediction's late label receives weight according to its prediction age.
Arrival timestamps must be nondecreasing. Prediction timestamps can arrive out
of order. `decay` and `half_life_seconds` are mutually exclusive; per-update
decay remains the default. Predicting at a future time updates the implied
shrinkage toward identity without mutating learned state. Pending labels remain
in the caller's system.

Install the `evaluation` extra (`pip install 'streamcal[evaluation]'`)
for River-scheduled delayed-label comparisons and
`arch` block-bootstrap intervals. Pass `prediction_times`, `label_delays`, and
`batch_size=1` to `compare_prequential`; `observe_mask` supports sparse labels.

## What is established

- Per-bin weighted counts and positive totals preserve the entire binned
  isotonic objective, including forgetting and the identity prior. Tests compare
  the fit to explicit weighted historical observations.
- Numerical arrays occupy exactly `40 * n_bins + 8` bytes: 808 bytes for 20 bins.
  Update work depends on batch size and bin count, not historical stream length.
- Known-truth Monte Carlo tests require calibration error to improve, protect
  already-calibrated inputs, and reject deliberately broken calibrators.
- A chronological Elec2 study compares validation-selected rolling, accumulating,
  frozen, and online references. Streamcal's Brier is 0.1778 versus 0.1787 for
  tuned rolling sigmoid; their paired interval includes zero. Its compact-state
  advantage holds even when latency rankings change; the report flags resource
  budgets that fail on the final period.

The compression proof concerns retained records, not fewer required labels.
Online parametric learners can use even smaller state. Those numbers describe
one stream; the report includes stationary, drifting, sparse, and delayed cases.
See the [formalization](https://finite-sample.github.io/streamcal/formalization.html)
for the objective identity, exact storage formula, and computation bounds.
See the [evidence and
limitations](https://finite-sample.github.io/streamcal/evidence.html) for the
full table, learning curve, environment, and reproduction command.

## Supported methods

| Method | Role | Retained state |
|---|---|---|
| `StreamingIsotonicCalibrator` | Adaptive monotone calibration | Fixed by `n_bins` |
| `BatchCalibrator(method="isotonic")` | scikit-learn isotonic | Full history or configured window |
| `BatchCalibrator(method="sigmoid")` | scikit-learn sigmoid | Full history or configured window |
| `BatchCalibrator(method="temperature")` | scikit-learn temperature | Full history or configured window |

The package is binary-only. Batch references expose `window_size` and
`refit_every`; `freeze()` retains the fitted map and discards raw history.
Their numerical implementations come from scikit-learn's public APIs. Sigmoid
uses raw probabilities as the classifier's scores; it is not the old
logistic-on-logits wrapper. A one-class window retains the last successful map
(or identity before readiness).

## Development

```bash
uv sync --all-groups
make ci
make evidence
```

`make ci-docker` runs the same checks in standard Python 3.12 and 3.14 images.

## License

MIT
