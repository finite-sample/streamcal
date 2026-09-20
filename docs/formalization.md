# Why compact state is sufficient

Fix B probability bins with centers c_b. Assign every prediction p_i to a bin
b(i), and let w_i be its nonnegative weight and y_i its binary outcome.
Fit monotone bin values z_1 ≤ ... ≤ z_B by minimizing

```text
L(z) = Σ_i w_i (y_i − z_b(i))² + λ Σ_b (z_b − c_b)².
```

Within a bin, define N_b = Σ w_i and S_b = Σ w_i y_i. Expanding the square gives

```text
Σ_{i in b} w_i (y_i − z_b)²
    = Σ_{i in b} w_i y_i² − 2 S_b z_b + N_b z_b².
```

The first term is constant with respect to the fitted values. Therefore the
minimizer depends on the raw observations only through N_b and S_b. Including
the identity prior gives a weighted isotonic problem with

```text
weight_b = N_b + λ
target_b = (S_b + λ c_b) / (N_b + λ).
```

Streamcal passes these weights and targets to scikit-learn's
`IsotonicRegression`. It does not implement the isotonic solver. Zero-weight
bins are omitted; their values are interpolated using the same upstream map.
Both the raw-history reference and the compact implementation must use this
same extension convention where the objective leaves values unidentified.

This establishes equality of the objective and its fitted map, up to floating
point arithmetic, for the **same bin assignments, weights, prior, and extension
convention**. It does not establish equality to unbinned isotonic regression.

## Exponential forgetting and late labels

For half-life h and prediction timestamp t_i, define the weight at current time
t as w_i(t) = 2^(-(t − t_i)/h). Only labels that have arrived by t enter the sums.

Advancing from t to t + Δ multiplies every existing weight by the same factor
2^(-Δ/h). Thus multiplying N_b and S_b by that factor updates the complete
historical objective without reading any historical observation. A newly
arriving label contributes its prediction-age weight, not weight one.

The identity prior remains fixed, so old evidence gradually loses influence
relative to identity. Prediction at a future time evaluates that objective on
temporary arrays and does not mutate learned state. All timestamps use one
caller-supplied clock in seconds. The package stores no pending-label queue.

At a fixed prediction time, partitioning the same labeled records into update
calls with the same final arrival time preserves the sufficient statistics.
Intermediate forecasts can differ when label availability differs. Per-update
decay deliberately has different semantics: changing the update cadence changes
the statistical weights.

## Storage and computation

The streaming calibrator owns five float64 arrays: B+1 edges and four B-length
arrays (centers, counts, positive mass, fitted values). Their storage is exactly

```text
8 × (5B + 1) = 40B + 8 bytes.
```

At B=20, this is 808 bytes, independent of historical observation count H.
Retaining float64 probabilities and labels requires 16H bytes before counting
the fitted model and Python/container overhead. A rolling reference requires
16W bytes for a window of W observations. Neither comparison says that the
entire Python process occupies 808 bytes. Scalar counters and interpreter
overhead are excluded; numerical array state is bounded, while an arbitrary
precision lifetime observation counter technically grows logarithmically.

For a new batch of m records, bin lookup uses O(m log B) comparisons, accumulation
and decay use O(m+B) work, and fitting uses only B aggregated records. A
conservative bound including upstream sorting is O(m log B + B log B) per
update. Temporary batch and solver arrays require O(m+B) space. This bound
does not grow with H.

An accumulating refit must at least read its H retained records per refit;
sorting them can cost O(H log H). A rolling refit instead depends on W, and
infrequent refits can reduce its amortized cost. An online parametric learner
can also have constant state and linear batch cost. The proof consequently
establishes independence from history length, not a universal speed advantage.

Ordinary calibration uses interpolation against B centers. Time-aware
calibration additionally refits the B-bin summary when evaluating elapsed time.
Neither prediction path depends on H.

## What “less data” means here

For this objective, raw historical records can be discarded without losing
fitting information. That is an exact compression result. It does not mean
fewer labels are required to learn the underlying calibration relationship.
Bin resolution, forgetting, and the prior change that statistical problem;
their quality trade-offs must be evaluated on future observations.
