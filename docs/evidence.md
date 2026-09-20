# Evidence: expected results and useful trade-offs

Correctness means obtaining the expected result for the stated objective.
The [formalization](formalization.md) proves that weighted bin counts and label
totals preserve the complete binned isotonic objective. Exact-reference tests
check its implementation with forgetting, the identity prior, late labels,
single-class data, and reset/serialization. This is a stronger foundation than
requiring the package to win an empirical benchmark.

When changing bins or forgetting, the objective itself changes. The useful
question is then which method gives the lowest chosen error within the user's
resource budget. `TradeoffReport.select` minimizes Brier, log loss, or binned
calibration error subject to p95 prediction/update latency and serialized-state
limits. The user supplies those limits; no universal equivalence margin is
assumed. A report can also return no feasible configuration.

## Protocol

The Elec2 dataset is [OpenML 151, version 1](https://www.openml.org/d/151), MD5
`8ca97867d960ae029ae3a9ac2c923d34`, with 45,312 chronological records.

1. Fit a frozen logistic base model on the first 20%.
2. Use the first half of the next 10% to initialize every candidate. Frozen
   references retain that fitted map; other methods continue updating.
3. Score the second half of that validation period in predict-then-observe
   order. Select one setting per predeclared family by Brier and select operating
   points by their stated objectives and budgets.
4. Reconstruct each selected candidate from the entire validation period, then
   evaluate it on the final 70%. No final-period labels select settings.

The fixed search includes 36 streamcal settings (20/50/100 bins,
0.5/0.9/0.99/1 decay, 1/10/100 prior weight); rolling references use
336/1,344/5,376-observation windows and refit every 1/4/16 batches. Online
logistic uses River SGD with learning rates 0.001/0.01/0.1. There are also raw,
frozen, and accumulating references. Each scored batch has up to 336 rows.
The complete search results are retained, including losing configurations.

Paired Brier intervals use `arch` circular block bootstrap over 1,344 consecutive
test observations and 2,000 replicates. This preserves pairing and weights
individual observations equally. These are descriptive dependence-aware
intervals from one stream; stationarity and block length affect their validity.
An interval containing zero does not prove equivalence.

## Interpreting the result

The stronger rolling references largely close the Brier gap from the original
accumulating-only comparison. Streamcal and tuned rolling sigmoid are close on
this stream, while streamcal retains less state. Direct rolling isotonic can
also be faster than streamcal in individual timing runs. Online logistic has
smaller serialized state than streamcal and wins some
validation selections, but performs worse on the final period. These results
show why an honest operating-point selector cannot always choose streamcal.
Validation-feasible choices can exceed a tight latency budget on the final
period; the generated table records those failures instead of treating the
validation measurement as a runtime guarantee.

Controlled experiments report distance to a known conditional probability,
alongside proper scores. They include cases where the raw model is already
correct and where adaptation adds estimation noise. Sparse-label experiments
still score all forecasts offline but reveal only a subset of labels for
updates. Delayed experiments use River's event scheduling and prediction-age
weights. Fixed settings across these scenarios expose weaknesses rather than
retuning each method on its test data.

The learning curves in the JSON show cumulative test-period scores. They do
not establish that fewer labels are needed: the methods saw a common warm-up
period, and the underlying distribution changes across the curve. The formal
claim is fewer **retained historical records** for the same binned objective.

## Reproduce

```bash
uv sync --all-groups --all-extras
make evidence
```

This writes `benchmarks/results/quality.json`, `benchmarks/results/resources.json`,
and the tables included below. Results carry package versions, dataset identity,
split points, settings, and a SHA-256 fingerprint of implementation files and
the dependency lock. The resource sweep runs methods serially in fresh Python
processes. Measurements describe this machine and invocation; neither p95
latency nor serialized size is a worst-case execution or process-memory bound.

The evidence workflow archives the JSON. Ordinary tests check numerical
identities and behavior; they do not require superiority on Elec2 or a timing
win on a shared CI runner.

```{include} evidence-results.md
```

## Limits and decision rules

- Binary probabilities only. Labels must correspond to the original forecasts.
- Per-update decay depends on batch cadence. Prediction-age half-life uses
  explicit timestamps and is invariant to regrouping available labels at a
  fixed final time, up to floating point arithmetic.
- Bin resolution changes the approximation. Equality to the binned objective
  does not imply equality to unbinned isotonic regression.
- Binned error depends on its bin count and can reward uninformative forecasts.
  Keep proper scores visible even when choosing that diagnostic as the objective.
- AUROC across a changing calibration map need not equal raw AUROC, even though
  every individual map is monotone.
- The calibrator owns no pending-label queue. The offline evaluator and caller's
  delayed-label storage are outside the bounded-state claim.
- Select on validation, evaluate once on a later period, and recheck resource
  budgets on the deployment hardware. Validation winners can lose out of sample.
