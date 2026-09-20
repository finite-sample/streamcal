"""Expected outcomes for scheduling, constrained selection and uncertainty."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from benchmarks.electricity import _moving_block_interval
from benchmarks.policies import RawReference
from benchmarks.suite import choose_families
from streamcal import BatchCalibrator, StreamingIsotonicCalibrator, compare_prequential

TRACE_LOGS = []


class Recorder(RawReference):
    def __init__(self):
        self.events = []
        TRACE_LOGS.append(self.events)

    def calibrate(self, probabilities):
        self.events.append(("predict", tuple(probabilities)))
        return super().calibrate(probabilities)

    def update(self, probabilities, outcomes):
        self.events.append(("update", tuple(probabilities)))
        return self


def test_composite_objective_selects_best_feasible_not_smallest():
    report = compare_prequential(
        [0.2, 0.8], [0, 1], {"raw": RawReference}, batch_size=1
    )
    template = report.results[0]
    slow = replace(
        template,
        name="slow",
        brier=0.01,
        binned_calibration_error=0.01,
        p95_predict_ms=101,
        serialized_bytes=100,
    )
    accurate = replace(
        template,
        name="accurate",
        brier=0.05,
        binned_calibration_error=0.03,
        p95_predict_ms=99,
        serialized_bytes=200,
    )
    tiny = replace(
        template,
        name="tiny",
        brier=0.1,
        binned_calibration_error=0.02,
        p95_predict_ms=1,
        serialized_bytes=50,
    )
    report = replace(report, results=(slow, accurate, tiny))
    assert report.select(max_predict_ms=100).name == "accurate"
    assert (
        report.select(objective="binned_calibration_error", max_predict_ms=100).name
        == "tiny"
    )
    assert report.select(max_predict_ms=100, max_serialized_bytes=100).name == "tiny"
    assert report.select(max_predict_ms=0.5) is None
    assert report.select(max_predict_ms=100, max_brier_degradation=0.01) is None
    assert {r.name for r in report.pareto_frontier()} == {"slow", "tiny"}
    with pytest.raises(ValueError, match="objective"):
        report.select(objective="ece")
    with pytest.raises(ValueError, match="max_predict_ms"):
        report.select(max_predict_ms=-1)


def test_delayed_event_order_has_expected_predictions():
    model = StreamingIsotonicCalibrator(n_bins=2, half_life_seconds=10, prior_weight=1)
    p = np.array([0.2, 0.8, 0.2])
    y = np.array([1, 0, 1])
    # First label arrives at t=2, second at t=1. Third forecast at t=3 sees both.
    expected = [0.2, 0.8]
    model.update([0.8], [0], prediction_times=[1], observed_at=1)
    model.update([0.2], [1], prediction_times=[0], observed_at=2)
    expected.append(model.calibrate([0.2], current_time=3)[0])
    report = compare_prequential(
        p,
        y,
        {"time": partial(StreamingIsotonicCalibrator, n_bins=2, half_life_seconds=10)},
        batch_size=1,
        prediction_times=[0, 1, 3],
        label_delays=[2, 0, 10],
    )
    assert np.allclose(report.results[0].squared_errors, (np.array(expected) - y) ** 2)


def test_methods_receive_identical_available_labels_and_sparse_mask():
    TRACE_LOGS.clear()
    factories = {"a": Recorder, "b": Recorder}
    report = compare_prequential(
        [0.2, 0.8],
        [0, 1],
        factories,
        batch_size=1,
        prediction_times=[0, 1],
        label_delays=[4, 0],
        observe_mask=[False, True],
    )
    assert report.results[0].squared_errors == report.results[1].squared_errors
    expected = [("predict", (0.2,)), ("predict", (0.8,)), ("update", (0.8,))]
    assert [expected, expected] == TRACE_LOGS


def test_paired_bootstrap_preserves_constant_expected_difference():
    report = compare_prequential(
        np.full(15, 0.5),
        np.zeros(15),
        {"a": RawReference, "b": RawReference},
        batch_size=4,
    )
    shifted = replace(
        report.results[1],
        squared_errors=tuple(np.asarray(report.results[1].squared_errors) + 0.1),
    )
    report = replace(report, results=(report.results[0], shifted))
    assert report.paired_interval(
        "b", "a", block_size=4, replicates=100
    ) == pytest.approx((0.1, 0.1, 0.1))


def test_partial_batch_bootstrap_uses_observation_weights():
    y = np.zeros(337)
    candidate = np.full(337, 0.5)
    candidate[-1] = 1
    reference = np.zeros(337)
    # A block spanning all two batches contains every observation on each draw.
    result = _moving_block_interval(
        y, candidate, reference, block_batches=2, replicates=100
    )
    expected = (336 * 0.25 + 1) / 337
    assert result["mean_brier_difference"] == pytest.approx(expected)
    assert result["ci_95_lower"] == pytest.approx(expected)
    assert result["ci_95_upper"] == pytest.approx(expected)


def test_validation_selection_depends_only_on_validation_report():
    report = compare_prequential(
        [0.2, 0.8],
        [0, 1],
        {"stream/a": RawReference, "stream/b": RawReference},
        batch_size=1,
    )
    report = replace(
        report,
        results=(
            replace(report.results[0], brier=0.2),
            replace(report.results[1], brier=0.1),
        ),
    )
    assert choose_families(report) == {"stream": "stream/b"}


def test_freezing_discards_history_and_leaves_predictions_fixed():
    model = BatchCalibrator().update([0.2, 0.8], [0, 1]).freeze()
    expected = model.calibrate([0.2, 0.8])
    model.update([0.2, 0.8], [1, 0])
    assert np.array_equal(model.calibrate([0.2, 0.8]), expected)
    assert model.history_bytes == 0
    with pytest.raises(ValueError, match="successful fit"):
        BatchCalibrator().freeze()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"configurations": {}}, "empty"),
        ({"configurations": {"": RawReference}}, "names"),
        ({"observe_mask": [1, 0]}, "boolean"),
        ({"label_delays": [0, 1]}, "requires"),
        ({"prediction_times": [1, 0]}, "ordered"),
        ({"prediction_times": [0, 1], "label_delays": [-1, 0]}, "nonnegative"),
        (
            {
                "configurations": {
                    "timed": partial(StreamingIsotonicCalibrator, half_life_seconds=1)
                }
            },
            "require",
        ),
    ],
)
def test_evaluation_rejects_ambiguous_inputs(kwargs, message):
    arguments = {"configurations": {"raw": RawReference}, "batch_size": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        compare_prequential([0.2, 0.8], [0, 1], **arguments)


def test_interval_and_update_budget_validation():
    report = compare_prequential(
        [0.2, 0.8], [0, 1], {"raw": RawReference}, batch_size=1
    )
    assert report.select(max_update_ns_per_observation=1e12) is not None
    with pytest.raises(ValueError, match="block_size"):
        report.paired_interval("raw", "raw", block_size=3)


def test_score_adapter_expected_classes_and_shapes():
    from streamcal.batch import ProbabilityClassifier

    adapter = ProbabilityClassifier()
    assert adapter.fit([[0.2]], [0]) is adapter
    assert np.array_equal(adapter.predict([[0.1], [0.9]]), [0, 1])
    assert np.array_equal(adapter.predict_proba([[0.0], [1.0]]), [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="shape"):
        adapter.predict_proba([0.1, 0.9])
