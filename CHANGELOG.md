# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Release tags match the version committed in `pyproject.toml`.

## [Unreleased]

## [0.2.0] - 2026-09-20

This release breaks the experimental 0.1 API. Predict with `calibrate` before
passing the original probabilities and available labels to `update`. Replace
method-specific batch classes with `BatchCalibrator(method=...)` and
`expected_calibration_error` with `binned_calibration_error`.

### Added

- Prediction-age half-life weighting with explicit label arrival timestamps.
- Constrained validation selection by Brier, log loss, or binned calibration
  error, with p95 call-latency and serialized-state budgets.
- An exact sufficient-statistic proof, rolling/frozen/online comparisons, and
  delayed-label evaluation using River's event scheduler.

- Leak-free prequential configuration comparison with quality, latency and
  fixed-state diagnostics.
- Reproducible known-truth and chronological Elec2 evidence for documented
  correctness and resource claims.
- Monte Carlo tests of calibration quality against known probabilities and
  mature batch references.

### Fixed

- Reject invalid probability, outcome, shape and parameter domains at public
  boundaries instead of broadcasting, clipping or returning NaNs.
- Weight streaming isotonic regression by decayed bin counts rather than giving
  every populated bucket equal influence.
- Delegate temperature fitting to scikit-learn's public calibration API.
- Use observation-weighted block-bootstrap intervals, including partial batches.
- Return no feasible choice for a zero-byte state budget instead of raising.

### Changed

- CI, docs and release now run on [py-canon](https://github.com/gojiplus/py-canon)'s
  reusable workflows in place of hand-rolled ones. The build backend moves from
  hatchling to `uv_build`, mypy is replaced by pyright, and the lint set widens
  to the fleet standard.
- `streamcal.__version__` is read from the installed distribution metadata
  rather than duplicated as a literal in `__init__.py`.
- Project URLs point at `finite-sample/streamcal`.
- The public contract is now predict-then-observe: call `calibrate` before
  `update`; `update` returns the calibrator and never fitted predictions for the
  same outcomes.
- Consolidate batch references into `BatchCalibrator(method=...)`, with rolling
  windows, refit cadence, and freezing. Sigmoid uses scikit-learn's probability
  input convention, replacing the former logistic-on-logits reference.
- Require scikit-learn 1.8 or newer. Optional delayed evaluation and intervals
  are available through `streamcal[evaluation]`.
- `expected_calibration_error` is now `binned_calibration_error` to make its
  estimator dependence explicit.

### Removed

- `NearlyIsotonicCalibrator`; the former fixed-step SGD routine did not
  implement the cited nearly-isotonic solution path.

- The `experiments/` and `ms/` trees that produced the original write-up. The
  package is now the library alone.

## [0.1.0] - 2026-03-27

Initial release: streaming isotonic and nearly-isotonic calibrators, batch
temperature/isotonic/Platt baselines, and Brier and expected-calibration-error
metrics.
