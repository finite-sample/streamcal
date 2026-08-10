"""Do the calibrators actually calibrate?

The package's headline claim is in its first paragraph: "Your model outputs 0.7
for a prediction. Is the true probability actually 70%? Usually not... You need
to learn a correction function." Nothing in ``tests/test_calibrators.py``
checked that the correction function corrects anything. Every assertion there is
about monotonicity, range, shape or rank, and the identity function satisfies
all four on already-monotone input -- so a calibrator that returned ``p_raw``
unchanged passed the entire suite.

The two statistical assertions the repo did have, ``assert ece < 0.05`` and
``assert ece > 0.5`` in ``test_metrics.py``, are hard-coded thresholds on a
binned estimator, and the first was computed from a single ``np.random.seed(42)``
draw. A number picked by hand cannot say whether an estimator is right; it can
only say whether it is roughly the size somebody expected once.

**The design.** The truth is set by construction. True probabilities ``p`` are
drawn, the reported probabilities are a deterministic strictly-monotone
distortion ``q = g(p)``, and outcomes are ``Bernoulli(p)``. Because ``g`` is a
bijection, the true calibration map is known in closed form -- ``E[y | q] =
g^-1(q)`` exactly -- so the calibration error of any map ``f`` is

    CE(f) = sqrt( E_q [ (f(q) - g^-1(q))^2 ] )

with no binning, no estimator bias, and no threshold to choose. It is evaluated
on a large fixed sample of ``q``, identical across replicates and calibrators,
so the Monte Carlo integration error is common to every number reported here.

**Why not ECE.** The plugin binned estimator is biased upward: it charges the
within-bin sampling noise of a *perfectly* calibrated forecaster as calibration
error, and the charge grows with the bin count. ``test_binned_ece_charges_a
_perfect_forecaster_for_noise`` measures that directly. A suite built on it would
be gating a quantity that depends on ``n_bins`` as much as on the forecaster.

The one binned quantity used here is the max-|Z| test below, and it is used
because its null distribution is *known* rather than assumed: see
``test_the_calibration_test_has_the_right_size_on_the_oracle_map``, which
measures its size at 0.0485 against a nominal 0.05 over 2000 replicates.

Every gate comes from `simcheck <https://github.com/finite-sample/simcheck>`_, so
its tolerance is derived from the replicate count. Every gate also has a test
that makes it fail: a constant calibrator and a shuffling calibrator are run
through the same assertions and required to be rejected. Without those, a suite
of passing assertions is not evidence that the assertions can fail.
"""

from __future__ import annotations

import functools
from statistics import NormalDist

import numpy as np
import pytest
from simcheck import assert_proportion, binomial_band, reps_for

from streamcal import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    PlattScaling,
    StreamingIsotonicCalibrator,
    TemperatureScaling,
    expected_calibration_error,
)

# Evaluation sample for the exact calibration error. Large enough that the Monte
# Carlo integration error is negligible beside the quantities being compared
# (raw CE 0.107, fitted CE 0.006-0.030), and fixed, so every replicate and every
# calibrator is scored against the same integral.
N_EVAL = 20_000

# Training stream: eight batches, because the streaming calibrators are built to
# be fed batches and their EMA needs more than one to mean anything.
N_TRAIN_BATCH = 500
N_BATCHES = 8

# Groups for the max-|Z| conditional calibration test.
GROUPS = 10
TEST_ALPHA = 0.05

# The two distortions. Both are strictly monotone, so both leave the ranking
# alone -- which is exactly why the existing rank-preservation and monotonicity
# tests cannot see them.
#
# "slope" is a pure temperature miscalibration, logit q = b logit p, so every
# calibrator here is correctly specified for it, including the one-parameter
# TemperatureScaling.
#
# "power" is q = p^gamma, which is not in the temperature family. It is included
# because a study where every method is well specified cannot tell a method that
# works from one that is being flattered by the fixture.
SLOPE = 0.5
POWER = 0.6

CALIBRATORS = {
    "StreamingIsotonic": lambda: StreamingIsotonicCalibrator(n_buckets=50, alpha=0.3),
    "NearlyIsotonic": lambda: NearlyIsotonicCalibrator(n_buckets=50, alpha=0.3),
    "Isotonic": IsotonicCalibrator,
    "Platt": PlattScaling,
    "Temperature": TemperatureScaling,
}


# ---------------------------------------------------------------------------
# The data generating process, whose inverse is the truth
# ---------------------------------------------------------------------------


def _logit(p):
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _distort(p, kind):
    """Turn true probabilities into the miscalibrated ones a model reports.

    Args:
        p: True probabilities in (0, 1).
        kind: ``"slope"``, ``"power"`` or ``"none"``.

    Returns:
        np.ndarray: Reported probabilities.

    Raises:
        ValueError: If ``kind`` is not one of the three.
    """
    if kind == "slope":
        return _sigmoid(SLOPE * _logit(p))
    if kind == "power":
        return p**POWER
    if kind == "none":
        return p
    raise ValueError(f"unknown distortion {kind!r}")


def _draw(rng, n, kind):
    """One sample of (reported probability, outcome).

    ``p`` is bounded away from 0 and 1 so that no bin of the max-|Z| test has a
    variance near zero, which would make its normal approximation the thing
    under test rather than the calibrator.

    Args:
        rng: Source of randomness.
        n: Sample size.
        kind: Which distortion to apply.

    Returns:
        tuple: ``(q, y)`` -- reported probabilities and Bernoulli outcomes.
    """
    p = rng.uniform(0.05, 0.95, n)
    return _distort(p, kind), (rng.random(n) < p).astype(float)


@functools.cache
def _evaluation_set(kind):
    """The fixed grid the calibration error is integrated over.

    Args:
        kind: Which distortion to apply.

    Returns:
        tuple: ``(p_true, q_reported)``, both length ``N_EVAL``.
    """
    rng = np.random.default_rng(12345)
    p = rng.uniform(0.05, 0.95, N_EVAL)
    return p, _distort(p, kind)


def _calibration_error(values, p_true):
    """Root mean squared distance from the known true calibration map.

    Args:
        values: What the map returned on the evaluation grid.
        p_true: The true conditional probabilities on that grid.

    Returns:
        float: The exact L2 calibration error.
    """
    return float(np.sqrt(np.mean((np.asarray(values) - p_true) ** 2)))


# ---------------------------------------------------------------------------
# Broken calibrators, so the gates have something to reject
# ---------------------------------------------------------------------------


class _ConstantCalibrator:
    """Returns the base rate for everything.

    The interesting negative control, because it is *calibrated in the large*:
    its mean output equals the mean outcome, so any check on E[p] - E[y] passes
    it. What it has destroyed is the resolution -- it answers the same thing for
    every input -- and only a conditional check sees that.
    """

    def __init__(self):
        self.rate = 0.5
        self._seen = []

    def update(self, p_raw, y):
        self._seen.append(np.asarray(y, dtype=float))
        self.rate = float(np.concatenate(self._seen).mean())
        return self.calibrate(p_raw)

    def calibrate(self, p_raw):
        return np.full(len(p_raw), self.rate)


class _ShufflingCalibrator:
    """Fits properly, then permutes the answers.

    Its output has exactly the marginal distribution of a correctly calibrated
    forecaster -- same values, same histogram, same range, same number of
    distinct levels -- and none of them are attached to the right input. Any
    assertion about the *set* of outputs passes it.
    """

    def __init__(self, seed=0):
        self.inner = IsotonicCalibrator()
        self.rng = np.random.default_rng(seed)

    def update(self, p_raw, y):
        self.inner.update(p_raw, y)
        return self.calibrate(p_raw)

    def calibrate(self, p_raw):
        out = np.asarray(self.inner.calibrate(p_raw)).copy()
        self.rng.shuffle(out)
        return out


BROKEN = {
    "constant": _ConstantCalibrator,
    "shuffling": _ShufflingCalibrator,
}


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


@functools.cache
def _study(kind, reps):
    """Fit every calibrator on ``reps`` independent streams and score each one.

    All calibrators see the *same* stream within a replicate, so the comparisons
    between them are paired and the differences reported are not stream noise.

    Args:
        kind: Which distortion the stream carries.
        reps: Number of replicates.

    Returns:
        dict: Name -> array of exact calibration errors, one per replicate. The
        key ``"raw"`` holds the uncalibrated input's error, which is a constant.
    """
    p_true, q_eval = _evaluation_set(kind)
    scores = {name: [] for name in list(CALIBRATORS) + list(BROKEN)}

    for replicate in range(reps):
        rng = np.random.default_rng(1000 + replicate)
        fitted = {name: factory() for name, factory in CALIBRATORS.items()}
        fitted.update({name: factory() for name, factory in BROKEN.items()})
        for _ in range(N_BATCHES):
            q, y = _draw(rng, N_TRAIN_BATCH, kind)
            for calibrator in fitted.values():
                calibrator.update(q, y)
        for name, calibrator in fitted.items():
            values = calibrator.calibrate(q_eval)
            scores[name].append(_calibration_error(values, p_true))

    out = {name: np.array(values) for name, values in scores.items()}
    out["raw"] = np.full(reps, _calibration_error(q_eval, p_true))
    return out


def _reps():
    """Replicate count for the current simcheck tier."""
    return reps_for()


# ---------------------------------------------------------------------------
# The conditional calibration test, and what makes it trustworthy
# ---------------------------------------------------------------------------

# Sidak, not Bonferroni: under the null the group statistics are independent, so
# P(no group flags) = P(|Z| < c)^GROUPS exactly, and the size is the nominal one
# rather than something conservative. Nothing here is chosen by hand -- c is
# whatever makes the overall size TEST_ALPHA.
_CRITICAL = NormalDist().inv_cdf((1.0 + (1.0 - TEST_ALPHA) ** (1.0 / GROUPS)) / 2.0)


def _group_index(q):
    """Equal-count groups of the *reported* probability.

    Grouping on ``q`` rather than on the calibrated output matters: ``q`` is
    fixed before any calibrator is fitted, so the groups are not chosen using
    the outcomes being tested, and a calibrator that collapses its output to a
    single value still lands in ten distinct groups.

    Args:
        q: Reported probabilities.

    Returns:
        np.ndarray: Group index in ``[0, GROUPS)`` for each element.
    """
    n = len(q)
    index = np.empty(n, dtype=int)
    index[np.argsort(q)] = np.minimum((np.arange(n) * GROUPS) // n, GROUPS - 1)
    return index


def _flags_miscalibration(values, y, groups):
    """Whether the max-|Z| test rejects perfect calibration of ``values``.

    Within each group the standardised score ``sum(y - f) / sqrt(sum f(1-f))``
    is asymptotically standard normal when ``f`` is the true conditional
    probability, by the Poisson-binomial CLT. That is the whole null: no
    parameters are estimated on this sample, so there is no degrees-of-freedom
    correction to get wrong.

    Args:
        values: The map's output on the sample.
        y: Outcomes.
        groups: Group index per observation.

    Returns:
        bool: True if any group's |Z| exceeds the critical value.
    """
    for group in range(GROUPS):
        mask = groups == group
        f = np.clip(np.asarray(values)[mask], 1e-9, 1.0 - 1e-9)
        z = (y[mask] - f).sum() / np.sqrt((f * (1.0 - f)).sum())
        if abs(z) > _CRITICAL:
            return True
    return False


# Fixed rather than taken from the tier. At the fast tier's 100 replicates the
# 3-sigma band around a nominal 0.05 is [0.000, 0.115], which would accept a test
# with twice its intended size; 400 replicates narrow it to [0.017, 0.083]. The
# study costs about five seconds.
SIZE_REPS = 400


def _rejection_rate(kind, use_oracle, reps=SIZE_REPS):
    """How often the max-|Z| test rejects, over independent fresh samples.

    Args:
        kind: Which distortion the sample carries.
        use_oracle: Score the exact inverse map (the truth) rather than the raw
            reported probabilities.
        reps: Number of replicates.

    Returns:
        float: Rejection rate.
    """
    rejections = 0
    for replicate in range(reps):
        rng = np.random.default_rng(7000 + replicate)
        p = rng.uniform(0.05, 0.95, N_EVAL)
        q = _distort(p, kind)
        y = (rng.random(N_EVAL) < p).astype(float)
        values = p if use_oracle else q
        rejections += _flags_miscalibration(values, y, _group_index(q))
    return rejections / reps


def test_the_calibration_test_has_the_right_size_on_the_oracle_map():
    """The gate on the gate: does the detector fire at the rate it claims?

    Handed the exact inverse map -- perfect calibration by construction -- the
    max-|Z| test must reject 5% of the time. If it rejected 20% of the time,
    every rejection recorded below would mean less than it appears to; if it
    rejected 0% of the time, every acceptance would mean nothing at all.

    Measured 0.0485 over 2000 replicates against a nominal 0.05, which is what
    licenses using a binned statistic in a file that otherwise refuses to.
    """
    rate = _rejection_rate("slope", use_oracle=True)
    assert_proportion(rate, SIZE_REPS, TEST_ALPHA, "max-|Z| size on the oracle map")


def test_already_calibrated_input_is_not_flagged():
    """A detector that fires on good input would make every result below moot.

    With no distortion the reported probabilities *are* the truth, so the test
    must treat them exactly as it treats the oracle map -- and it does, because
    they are the same array.
    """
    rate = _rejection_rate("none", use_oracle=False)
    assert_proportion(rate, SIZE_REPS, TEST_ALPHA, "max-|Z| on undistorted input")


@pytest.mark.parametrize("kind", ["slope", "power"])
def test_the_miscalibration_is_real_and_detectable(kind):
    """The fixture check. Without it the reduction studies prove nothing.

    If the distortion were too small to detect, every calibrator would score well
    by doing nothing, and the whole file would be measuring noise.

    Args:
        kind: Which distortion to check.
    """
    rate = _rejection_rate(kind, use_oracle=False)
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_proportion(rate, SIZE_REPS, TEST_ALPHA, f"raw {kind} input")

    _, high = binomial_band(TEST_ALPHA, SIZE_REPS)
    assert rate > high, f"{kind}: raw input rejected only {rate:.3f} of the time"


# ---------------------------------------------------------------------------
# The headline claim: calibration reduces calibration error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["slope", "power"])
@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibration_reduces_the_true_calibration_error(name, kind):
    """Every calibrator must beat the input it was handed, in every replicate.

    The nominal rate is 1.0, so the binomial band collapses to a point and this
    asks for improvement every single time rather than on average. That is the
    right claim here because the margin is not marginal: the distortion costs
    0.1067 (slope) or 0.1430 (power) of calibration error, and the fitted maps
    average between 0.0049 and 0.0300, with no replicate above 0.0460. A
    replicate that failed to improve at those distances would be a defect, not
    bad luck.

    The one exception is written down rather than smoothed over.
    TemperatureScaling under the power distortion reduces the error from 0.1430
    to 0.1381 -- it improves in every replicate, so this passes, but it removes
    3.4% of the miscalibration where the others remove 79% to 92%. It has one
    parameter and the distortion is not in its family.
    ``test_temperature_scaling_barely_helps_outside_its_family`` pins that gap so
    it cannot silently turn into a claim.

    Args:
        name: Which calibrator.
        kind: Which distortion.
    """
    reps = _reps()
    scores = _study(kind, reps)
    improved = float(np.mean(scores[name] < scores["raw"]))
    assert_proportion(improved, reps, 1.0, f"{name} on {kind}, error reduced")


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibration_does_not_damage_already_calibrated_input(name):
    """The case a bare "reduces ECE" test would wave through.

    On input that is already correct there is nothing to gain, so a calibrator
    can only do harm, and the harm has to be bounded by something that is not a
    number somebody picked. Two yardsticks, both set by the data generating
    process:

    **It must still beat the best constant forecaster.** The lowest calibration
    error reachable by ignoring the input entirely is sd(p) = 0.2597. A
    calibrator that has collapsed its output, which is the classic way to destroy
    good input, cannot clear this.

    **The damage must be smaller than the miscalibration it exists to remove.**
    Fitting on already-calibrated data costs between 0.0079 and 0.0300 of
    calibration error on average, and 0.0460 in the worst replicate, against the
    0.1067 the slope distortion costs. Handing this package clean data and
    letting it fit is a net loss of a few thousandths.

    Args:
        name: Which calibrator.
    """
    reps = _reps()
    clean = _study("none", reps)
    p_true, distorted_q = _evaluation_set("slope")
    miscalibration_cost = _calibration_error(distorted_q, p_true)
    constant_floor = float(np.std(_evaluation_set("none")[0]))

    beats_constant = float(np.mean(clean[name] < constant_floor))
    assert_proportion(beats_constant, reps, 1.0, f"{name} keeps resolution")

    below_the_distortion = float(np.mean(clean[name] < miscalibration_cost))
    assert_proportion(
        below_the_distortion, reps, 1.0, f"{name} damage under the distortion"
    )


def test_temperature_scaling_barely_helps_outside_its_family():
    """A measured limitation, pinned so it cannot drift into a claim.

    TemperatureScaling has one parameter and the power distortion is not a
    temperature shift, so it can only trade one miscalibration for a smaller one.
    It removes 3.4% of the error, against 91.5% for Platt, 80.2% for isotonic and
    79.4% for the streaming isotonic calibrator on the same streams. Nothing here
    is broken -- the README lists it as a batch baseline -- but "it reduces
    calibration error" is a true sentence that would badly misdescribe it, which
    is why the reduction test above carries this footnote.

    The comparison is relative rather than against a threshold: temperature
    scaling must remove less than half of what the *weakest* other calibrator
    removes. That stays meaningful if the distortion is ever retuned.
    """
    reps = _reps()
    scores = _study("power", reps)
    raw = scores["raw"][0]
    removed = {
        name: 1.0 - float(np.mean(values)) / raw
        for name in CALIBRATORS
        for values in [scores[name]]
    }
    others = [share for name, share in removed.items() if name != "Temperature"]
    assert removed["Temperature"] < 0.5 * min(others), removed


# ---------------------------------------------------------------------------
# Negative controls: the same gates, run against calibrators that are wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["slope", "power"])
@pytest.mark.parametrize("name", sorted(BROKEN))
def test_a_broken_calibrator_is_caught_not_reducing_the_error(name, kind):
    """The reduction gate must reject a calibrator that does not calibrate.

    Both controls are built to survive the assertions the suite already had. The
    constant one is monotone (weakly), in range, right-shaped and rank
    preserving; the shuffling one has the exact output distribution of a correct
    calibrator. ``tests/test_calibrators.py`` would pass either of them.

    Args:
        name: Which broken calibrator.
        kind: Which distortion.
    """
    reps = _reps()
    scores = _study(kind, reps)
    improved = float(np.mean(scores[name] < scores["raw"]))
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_proportion(improved, reps, 1.0, f"{name} on {kind}")

    assert improved == 0.0, f"{name} improved on {improved:.0%} of replicates"


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_a_broken_calibrator_is_caught_damaging_clean_input(name):
    """And the damage gate must reject them too.

    This is the half that matters: a calibrator can look fine on miscalibrated
    input -- anything beats a large distortion -- and still be destroying the
    signal it is handed. Both controls fail the constant-forecaster floor.

    Args:
        name: Which broken calibrator.
    """
    reps = _reps()
    clean = _study("none", reps)
    constant_floor = float(np.std(_evaluation_set("none")[0]))
    beats_constant = float(np.mean(clean[name] < constant_floor))
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_proportion(beats_constant, reps, 1.0, f"{name} keeps resolution")


def test_the_constant_calibrator_passes_a_calibration_in_the_large_check():
    """Why the conditional test is the one being run, in one measurement.

    E[f] - E[y] is the quantity a mean-calibration check looks at, and the
    constant calibrator drives it to zero by definition: it reports the base
    rate. It is, in the large, perfectly calibrated. It is also useless.

    The max-|Z| test rejects it every time, because within the lowest tenth of
    reported probabilities the outcome rate is nowhere near the base rate. That
    difference is the entire reason this file scores conditional error rather
    than a mean.
    """
    rng = np.random.default_rng(4242)
    calibrator = _ConstantCalibrator()
    for _ in range(N_BATCHES):
        q, y = _draw(rng, N_TRAIN_BATCH, "slope")
        calibrator.update(q, y)

    q, y = _draw(rng, N_EVAL, "slope")
    values = calibrator.calibrate(q)

    in_the_large = abs(float(values.mean()) - float(y.mean()))
    assert in_the_large < 0.01, in_the_large

    assert _flags_miscalibration(values, y, _group_index(q))


# ---------------------------------------------------------------------------
# Why the binned estimator is not the yardstick
# ---------------------------------------------------------------------------


def test_binned_ece_charges_a_perfect_forecaster_for_noise():
    """``expected_calibration_error`` is not zero on a perfectly calibrated one.

    The plugin estimator compares each bin's outcome rate to its mean forecast,
    and the outcome rate carries binomial noise of order 1/sqrt(bin count). It
    charges that noise as calibration error, so the reported number rises with
    the bin count while the forecaster is unchanged -- 0.0104 at 5 bins, 0.0144
    at 20, 0.0481 at 200, on a forecaster whose true calibration error is exactly
    zero.

    This is a property of the estimator, not a defect in this package, and it is
    pinned here for one reason: ``assert ece < 0.05`` cannot distinguish a
    calibrated forecaster from a miscalibrated one without also fixing n and
    n_bins, and no test in this repo states either. The studies above use the
    exact error instead.
    """
    rng = np.random.default_rng(999)
    p = rng.uniform(0.05, 0.95, 10_000)
    y = (rng.random(10_000) < p).astype(float)

    coarse = expected_calibration_error(y, p, n_bins=5)
    fine = expected_calibration_error(y, p, n_bins=200)

    assert coarse > 0.0
    assert fine > 3.0 * coarse, (coarse, fine)
