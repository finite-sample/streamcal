# Why compact state is sufficient

This page shows that the streaming calibrator can discard every past
observation and keep only two numbers per bin without changing what it fits.
[Scope](#scope) says exactly what that result covers and what it does not.

## Setup

Split $[0, 1]$ into $B$ equal-width bins with centers $c_1 < \dots < c_B$.
Each labeled prediction $i$ has a raw probability $p_i$, a binary outcome
$y_i \in \{0, 1\}$, a nonnegative weight $w_i$, and falls in bin $b(i)$.

The calibrator fits one value per bin, $z_1 \le z_2 \le \dots \le z_B$, by
minimizing squared error plus a pull of strength $\lambda \ge 0$ (the
`prior_weight`) toward the identity map:

$$
L(z) = \sum_i w_i \bigl(y_i - z_{b(i)}\bigr)^2
     + \lambda \sum_{b=1}^{B} \bigl(z_b - c_b\bigr)^2 .
$$

## Two numbers per bin are enough

For each bin, let

$$
N_b = \sum_{i \in b} w_i
\qquad\text{and}\qquad
S_b = \sum_{i \in b} w_i y_i
$$

be the weighted count and the weighted number of positives. Expanding the
square for one bin gives

$$
\sum_{i \in b} w_i \bigl(y_i - z_b\bigr)^2
  = \underbrace{\sum_{i \in b} w_i y_i^2}_{\text{does not depend on } z}
    - 2 S_b z_b + N_b z_b^2 .
$$

The first term does not affect which $z$ minimizes $L$, so the minimizer
depends on the raw observations only through $N_b$ and $S_b$. Adding the prior
term and completing the square turns the whole problem into a weighted
isotonic regression over the $B$ bins, with

$$
\text{weight}_b = N_b + \lambda,
\qquad
\text{target}_b = \frac{S_b + \lambda c_b}{N_b + \lambda}.
$$

Each target is the bin's observed positive rate, shrunk toward the bin center.
Streamcal passes these weights and targets to scikit-learn's
`IsotonicRegression`; it does not implement the isotonic solver. A bin with
zero weight (possible only when $\lambda = 0$) is left out of the fit, and its
value is interpolated from its neighbours. A raw-history implementation has to
use the same convention for the two to agree, because the objective leaves
those values undetermined.

To calibrate a new probability, streamcal linearly interpolates the fitted
values $z_b$ between bin centers. Below $c_1$ it returns $z_1$, and above
$c_B$ it returns $z_B$.

## Forgetting old labels

The weights $w_i$ are how streamcal forgets. There are two modes.

**Half-life mode** (`half_life_seconds` $= h$). A label for a prediction made
at time $t_i$ has weight

$$
w_i(t) = 2^{-(t - t_i)/h}
$$

at the current time $t$, and only labels that have arrived by $t$ count.
Moving forward by $\Delta$ multiplies every weight by the same factor:

$$
w_i(t + \Delta) = 2^{-\Delta/h}\, w_i(t).
$$

So multiplying $N_b$ and $S_b$ by $2^{-\Delta/h}$ gives exactly the sums for
the new time, without revisiting any past observation. A late label enters with
its age-based weight, not with weight one. Calibrating at a later time does the
same rescaling on temporary copies and leaves the stored state unchanged. All
timestamps are seconds on one caller-supplied clock, and the package keeps no
queue of pending labels.

**Per-update mode** (`decay` $= d \in [0, 1]$). Each call to `update` first
multiplies $N_b$ and $S_b$ by $d$ and then adds the new batch with weight one.
A batch observed $k$ updates ago therefore has weight $d^{k}$. This counts
update calls, not elapsed time, so splitting the same labels into more or fewer
calls changes their weights. Setting $d = 1$ keeps everything.

In both modes the prior strength $\lambda$ does not decay. As evidence ages, a
bin's target drifts back toward its center, that is, toward leaving the raw
probability unchanged.

## Storage and computation

The calibrator's numerical state is five float64 arrays: $B + 1$ bin edges and
four length-$B$ arrays (centers, $N_b$, $S_b$, fitted values). That is exactly

$$
8 \times (5B + 1) = 40B + 8 \ \text{bytes},
$$

or 808 bytes at $B = 20$, however many observations $H$ it has seen. Keeping
the raw history instead takes $16H$ bytes (a float64 probability and label per
observation), before counting the fitted model, and a rolling window of $W$
observations takes $16W$ bytes. These figures cover the numerical arrays only,
not Python objects or the interpreter.

An update with a batch of $m$ labels costs

| Step | Work |
|---|---|
| Find each label's bin | $O(m \log B)$ |
| Decay and accumulate $N_b$, $S_b$ | $O(m + B)$ |
| Refit on $B$ aggregated bins | $O(B \log B)$ |

so $O(m \log B + B \log B)$ time and $O(m + B)$ temporary memory in total.
Neither depends on $H$. Calibrating interpolates against $B$ centers; in
half-life mode it also refits the $B$ bins for the elapsed time.

By comparison, refitting on the full history reads all $H$ records each time,
and sorting them costs $O(H \log H)$. A rolling window costs depend on $W$
instead. An online parametric model can also have constant state. The result
here is that cost does not grow with history length, not that streamcal is
faster than every alternative.

(scope)=
## Scope

### What is proved

The results above are exact algebra, up to floating-point rounding.

1. **Sufficiency.** For a fixed set of bins, weights, and $\lambda$, fitting
   from $(N_b, S_b)$ gives the same bin values as fitting from the full labeled
   history.
2. **Forgetting.** Rescaling $(N_b, S_b)$ reproduces the half-life weights
   exactly, including for late labels. Per-update mode applies the geometric
   weights $d^k$ described above.
3. **Resources.** State size and per-update cost do not depend on the number
   of observations seen.

### What the results depend on

- **Binary outcomes and raw inputs.** Outcomes are 0 or 1, probabilities lie in
  $[0, 1]$, and `update` receives the *original* model probabilities, the same
  ones that were binned, not calibrated ones.
- **The binned objective.** Every prediction in a bin shares one fitted value,
  which is assigned to the bin center. The equivalence is with *this*
  objective, not with isotonic regression on unbinned probabilities.
- **A monotone map.** The fit is constrained to be nondecreasing in $p$. If the
  true relationship between raw probability and outcome rate is not monotone,
  the result is the closest monotone map under the loss, not the true map.
- **One clock, in order (half-life mode).** Label arrival times do not go
  backwards across updates, and no prediction time is later than its label's
  arrival time. The calibrator raises an error otherwise.
- **Forgetting weights only.** No other per-observation weights (for example,
  importance weights) enter $N_b$ and $S_b$.

### What is not claimed

- **Equality with unbinned isotonic regression.** Binning loses whatever
  variation the true map has within a bin; more bins reduce that loss but leave
  fewer labels per bin.
- **Statistical guarantees.** Nothing here proves consistency, a convergence
  rate, a bound on calibration error, or a regret bound. How well the fitted
  map calibrates future predictions is an empirical question, measured on the
  [evidence](evidence.md) page.
- **Tracking drift.** Forgetting down-weights old labels but does not
  guarantee that the fit follows any particular kind of change. The fixed prior
  also pulls sparsely observed bins toward the raw probabilities.
- **Robustness to which labels arrive.** Only labels that have arrived enter
  the sums. If whether or when a label arrives depends on the outcome (say,
  positives are confirmed faster than negatives), the observed rates are
  biased, and nothing in this construction corrects for that.
- **Needing fewer labels.** Discarding raw records loses no *fitting*
  information for this objective. That is a compression result. It does not
  mean fewer labels are needed to learn the calibration relationship; bin
  width, forgetting, and the prior all change that statistical problem.
