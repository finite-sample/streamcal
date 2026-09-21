"""Tests for leak-free prequential trade-off evaluation."""

import numpy as np
import pytest

from streamcal import StreamingIsotonicConfig, compare_prequential


def test_compare_prequential_reports_quality_cost_and_trajectories():
    probabilities = np.tile(np.array([0.1, 0.3, 0.7, 0.9]), 20)
    outcomes = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 20)

    report = compare_prequential(
        probabilities,
        outcomes,
        {
            "stable": StreamingIsotonicConfig(n_bins=4, decay=1.0),
            "adaptive": StreamingIsotonicConfig(n_bins=4, decay=0.5),
        },
        batch_size=8,
    )

    assert report.n_observations == 80
    assert report.batch_size == 8
    assert [result.name for result in report.results] == ["stable", "adaptive"]
    for result in report.results:
        assert len(result.batch_brier) == 10
        assert len(result.cumulative_brier) == 10
        assert 0.0 <= result.brier <= 1.0
        assert result.log_loss >= 0.0
        assert result.predict_ns_per_observation >= 0.0
        assert result.update_ns_per_observation >= 0.0
        assert result.state_bytes > 0


def test_compare_prequential_first_batch_is_identity_for_every_configuration():
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])

    report = compare_prequential(
        probabilities,
        outcomes,
        {"candidate": StreamingIsotonicConfig(n_bins=4, decay=0.5)},
        batch_size=4,
    )

    assert report.results[0].batch_brier == pytest.approx(
        (float(np.mean((probabilities - outcomes) ** 2)),)
    )


def test_compare_prequential_validates_batch_size():
    with np.testing.assert_raises_regex(ValueError, "batch_size"):
        compare_prequential(
            np.array([0.2, 0.8]),
            np.array([0.0, 1.0]),
            {"candidate": StreamingIsotonicConfig()},
            batch_size=0,
        )
