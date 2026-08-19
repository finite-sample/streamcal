# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Release tags match the version committed in `pyproject.toml`.

## [Unreleased]

### Changed

- CI, docs and release now run on [py-canon](https://github.com/gojiplus/py-canon)'s
  reusable workflows in place of hand-rolled ones. The build backend moves from
  hatchling to `uv_build`, mypy is replaced by pyright, and the lint set widens
  to the fleet standard.
- `streamcal.__version__` is read from the installed distribution metadata
  rather than duplicated as a literal in `__init__.py`.
- Project URLs point at `finite-sample/streamcal`. The metadata on PyPI still
  names the pre-move `soodoku/mw-calibration` and only refreshes on the next
  release.

## [0.2.0] - unreleased

### Added

- Monte Carlo tests that check the calibrators actually reduce calibration
  error, and a study covering the batch baselines.

### Removed

- The `experiments/` and `ms/` trees that produced the original write-up. The
  package is now the library alone.

## [0.1.0] - 2026-03-27

Initial release: streaming isotonic and nearly-isotonic calibrators, batch
temperature/isotonic/Platt baselines, and Brier and expected-calibration-error
metrics.
