# Correctness audit

## Findings resolved for the next release

| Finding | Consequence | Resolution |
|---|---|---|
| `update` returned predictions after learning their own outcomes | Easy in-sample leakage | Separate `calibrate` from `update`; test outcome independence |
| NumPy broadcasting accepted unequal metric inputs | Plausible but wrong scores | Central equal-length validation |
| Probabilities, outcomes, and tuning parameters lacked domain checks | Silent clipping, NaNs, or downstream errors | Validate every public boundary |
| Streaming PAV weighted every bin equally | Sparse and dense bins influenced the fit equally | Weighted sufficient statistics and weighted PAV |
| EMA averaged batch rates without their counts | Results depended on arbitrary batch composition | Decayed counts and positive mass |
| Nearly-isotonic SGD was not the cited solution-path algorithm | The name overstated method fidelity | Remove it rather than preserve an unsupported API |
| Temperature scaling duplicated an established method | Extra numerical implementation to maintain | Delegate to scikit-learn's public temperature calibration |
| ECE name hid estimator dependence on bins and sample size | Encouraged overinterpretation | Rename it and document it as a binned diagnostic |
| PyPI 0.1 and `main` exposed different algorithm families | Install instructions could imply the wrong API | Mark the unreleased API and add migration notes |

## Supported contracts

- `StreamingIsotonicCalibrator` estimates a binary monotone calibration map
  from fixed-width bin sufficient statistics with optional exponential
  forgetting and identity shrinkage.
- Batch references delegate to scikit-learn and expose accumulating, rolling,
  infrequent-refit, and frozen policies. Sigmoid uses scikit-learn's probability
  score convention, replacing the old logistic-on-logits wrapper.
- Proper scores evaluate forecast usefulness. Calibration diagnostics are not
  used alone because a constant base-rate forecast can appear marginally
  calibrated while destroying resolution.

The implementation is compared with scikit-learn's official isotonic and
logistic APIs. Temperature scaling follows the one-parameter logit scaling
studied by [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).
The removed nearly-isotonic name referred to [Tibshirani, Höfling, and
Tibshirani (2011)](https://doi.org/10.1198/TECH.2010.10111), whose path algorithm
was not implemented by the former fixed-step routine.

## Implementation inventory

| Operation | Implementation | Streamcal responsibility |
|---|---|---|
| Weighted isotonic optimization | `sklearn.isotonic.IsotonicRegression` | Accumulate the exact binned sufficient statistics |
| Batch isotonic | `sklearn.isotonic.IsotonicRegression` | Retained window, refit cadence, freezing |
| Batch sigmoid, temperature | `sklearn.calibration.CalibratedClassifierCV` and `FrozenEstimator` | Score adapter and the same history/refit policies |
| Brier, log loss, AUROC | `sklearn.metrics` | Strict aligned binary input validation |
| Calibration curve | `sklearn.calibration.calibration_curve` | Weight upstream bin gaps by matching bin counts |
| Interpolation, bin accumulation | NumPy | Define fixed-width bins and state semantics |
| Paired dependent-data intervals | `arch.bootstrap.CircularBlockBootstrap` | Paired losses and observation-weighted statistic |
| Delayed label order | `river.stream.simulate_qa` | Forward original probability and timestamps |
| Online logistic benchmark | River logistic regression and SGD | Translate one probability into a logit feature |
| Peak process memory | Standard `resource` and `psutil` | Run isolated measurement processes |

The frozen score adapter has no learned base model. Its one-fold splitter
supplies all available rows for calibration and stores no row-index history.
Sharing these train/test indices is safe only because the adapter is stateless
and frozen; base-model training occurs on an earlier, disjoint period.

The custom mathematical behavior is the summary-state recurrence, prediction-age
weighting, and fixed identity prior. Their objective and equivalence conditions
are proved in [the formalization](formalization.md). Retention, scheduling
adapters, and budget selection are integration policies, not new solvers.

Input-boundary and exactly computable expected-result tests are the primary
correctness checks. Known-truth simulations and held-out comparisons assess the
statistical choices. Latency budgets use observed p95 whole-call times; they are
measurements, not worst-case execution guarantees.
