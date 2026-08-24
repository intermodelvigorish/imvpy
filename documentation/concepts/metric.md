# Metric Definition

IMV compares two systems that assign probabilities to a binary outcome. The
implementation follows four transformations.

## 1. Observation likelihood

For outcome `y_i` in `{0, 1}` and predicted positive-class probability `p_i`, the
probability assigned to what occurred is:

```text
L_i = p_i**y_i * (1 - p_i)**(1 - y_i)
```

Exact zero and one probabilities are clipped to `[epsilon, 1 - epsilon]`; the
default is `epsilon=1e-9`.

## 2. Geometric mean likelihood

`ll(y, p)` aggregates the per-observation likelihoods on the log scale:

```text
a = exp(mean(y * log(p) + (1 - y) * log(1 - p)))
```

This is a geometric mean Bernoulli likelihood in `(0, 1]`, not a conventional
sum log-likelihood. Larger values mean the predictor assigned more probability
to the observed outcomes.

## 3. Equivalent-coin information weight

`get_w(a)` finds the `w >= 0.5` branch satisfying:

```text
w * log(w) + (1 - w) * log(1 - w) = log(a)
```

The left side is negative binary entropy. It increases monotonically from
`-log(2)` at `w=0.5` toward `0` as `w` approaches one, so every `a` in `[0.5, 1]`
has one root on this branch. `w=0.5` represents chance-level information;
larger `w` represents a more informative equivalent coin.

The default `brentq` backend brackets this monotone root. The legacy `lbfgsb`
backend remains available for parity with older analyses but is slower and can
stall at its lower bound. See [Compatibility](../reference/compatibility.md).

## 4. Relative change

For baseline weight `w_basic` and enhanced weight `w_enhanced`:

```text
IMV(basic -> enhanced) = (w_enhanced - w_basic) / w_basic
```

A positive score means the enhanced predictions have a larger transformed
likelihood; zero means equal weights; a negative score means the enhanced
predictions score worse.

## Directionality

IMV is directional because the baseline weight is the denominator:

```text
IMV(B -> E) != -IMV(E -> B)
```

Consequently, magnitudes calculated against different baselines are not directly
comparable. An ablation matrix is generally neither symmetric nor
antisymmetric. The multiclass pairwise matrix is symmetric for a different
reason: swapping class labels complements both outcomes and renormalized
probabilities, leaving `ll` unchanged without reversing the model roles.

## The chance floor

No real equivalent-coin root exists when `a < 0.5`. Finite held-out samples can
put a fitted null model slightly below the floor, so `get_w` uses a documented
boundary approximation when the residual `abs(log(2a))` is at most 0.5 nats.
Further below the floor it emits `BelowChanceLikelihoodWarning` and returns
`NaN` instead of inventing a weight.

Use `information_deficit(a) = log(2a)` to report the distance from chance. It is
zero at `a=0.5`, positive above chance, and negative below chance.

## Numerical defaults

| Setting | Default | Effect |
|---|---:|---|
| Probability clipping `epsilon` | `1e-9` | Keeps logarithms finite |
| Inverse method | `brentq` | Bracketed monotone root finding |
| Weight bounds | `[0.5, 1 - 1e-12]` | Selects the published upper branch and avoids `0 * log(0)` |
| Below-chance tolerance | `0.5` nats | Boundary approximation for small finite-sample deficits |
| Legacy optimizer tolerance | `1e-9` | Used only by `method="lbfgsb"` |

`config/settings.yaml` records these defaults, and contract tests compare them
to the live Python signatures.

