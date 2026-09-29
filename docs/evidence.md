# Evidence

There are two questions: does streamcal compute what it claims, and is that
useful?

**Does it compute what it claims?** The [formalization](formalization.md)
proves that keeping two numbers per bin fits exactly the same map as keeping
the whole history. The test suite checks the implementation against a
full-history reference, including forgetting, the prior, late labels,
streams with only one outcome class, and reset and serialization. That is the
correctness guarantee. It does not depend on winning any benchmark.

**Is it useful?** That is an empirical question, and this page answers it on
one real stream and a set of simulations.

## What we found

- **On the electricity stream, streamcal roughly halved forecast error**
  (Brier score down 51% from the uncalibrated model). It was clearly better
  than 9 of the 11 alternatives. Refitting sigmoid or isotonic calibration on
  the most recent 336 labels did about as well: the data cannot separate
  them from streamcal. Streamcal stored about a fifth of their state, and
  under 1/400th of what the methods that refit on all history stored.
- **In simulations where the true probability is known**, streamcal had the
  smallest error of the three adaptive methods in every scenario. When the
  model was already calibrated, every adaptive method made it slightly worse,
  streamcal by the least (about 5 percentage points of noise).
- **Choosing by validation is not a guarantee.** Under a 1 ms update budget,
  the streamcal setting chosen on validation data took 1.01 ms on the test
  period, just over budget. Under a looser budget, online logistic regression
  won on validation but did worse than streamcal on test.

```{include} evidence-results.md
```

## How the comparison was run

**Data.** The Elec2 electricity-price dataset ([OpenML 151, version
1](https://www.openml.org/d/151), MD5 `8ca97867d960ae029ae3a9ac2c923d34`):
45,312 half-hourly records, used in time order.

1. **Base model.** A logistic regression is fit on the first 20% and then
   frozen. Its probabilities are what every method calibrates.
2. **Validation.** Over the next 10%, each method warms up on the first half.
   Its settings are then chosen on the second half by Brier score.
3. **Test.** The final 70% is never used to choose anything. Every method
   predicts each batch of 336 records (one week) before seeing that batch's
   outcomes, then learns from them.

**Settings searched.** Streamcal: 36 combinations of 20/50/100 bins,
0.5/0.9/0.99/1 decay and 1/10/100 prior weight. Rolling methods: windows of
336, 1,344 or 5,376 records, refit every 1, 4 or 16 batches. Online logistic
regression (River): learning rates 0.001, 0.01 or 0.1. Frozen, full-history
and uncalibrated references need no tuning. The JSON results keep every
setting, including the ones that lost.

**Intervals.** Paired differences are resampled in blocks of 1,344
consecutive records (2,000 resamples), so the intervals account for
neighbouring records being alike. They describe this one stream and assume
its behaviour is roughly stable across blocks.

## Caveats

- **Binary outcomes only.** Always update with the model's original
  probabilities, not calibrated ones.
- **Per-update decay depends on batching.** Splitting the same labels into
  more update calls forgets faster. Half-life mode uses timestamps instead and
  is unaffected by how labels are grouped.
- **Binned calibration error depends on the bin count** and can reward a
  forecast that ignores its input. Look at Brier score alongside it.
- **Ranking can change.** Each fitted map preserves order, but the map changes
  over time, so AUROC over a whole stream can differ from the raw model's.
- **Late labels are your storage.** Streamcal keeps no queue of pending
  labels. Holding them until they arrive is the caller's job and is outside
  the fixed-memory claim.
- **Timings are from one machine** (an Apple-silicon Mac). Re-check budgets on
  your own hardware.
- **Not evidence of needing fewer labels.** The learning curves in the JSON are
  cumulative scores on a changing stream. The formal result is about storing
  fewer past records, not learning from fewer labels.

## Reproduce

The numbers above were measured at
[commit 492bd39](https://github.com/finite-sample/streamcal/tree/492bd3986bab52dccc04d14f4d3a80a0a8ce096d).
The code has changed since, so measuring again gives new numbers:

```bash
uv sync --all-groups --all-extras
make evidence
```

This rewrites `benchmarks/results/quality.json`, `benchmarks/results/resources.json`
and the tables on this page. The JSON records package versions, dataset
identity, split points, settings, and a fingerprint of the source and
dependency lock. The test suite does not require streamcal to win on Elec2 or
to beat any timing on CI.
