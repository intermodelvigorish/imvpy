# Vanilla IMV

Use vanilla IMV when two binary probabilistic predictors have been evaluated on
the same outcomes.

## Probability mode

```python
import numpy as np
import pandas as pd

from imv import vanilla_imv

outcomes = pd.Series([1, 0, 1, 0, 1, 0])
baseline = 0.5
enhanced = np.array([0.82, 0.21, 0.75, 0.18, 0.68, 0.31])

score = vanilla_imv(baseline, enhanced, outcomes)
```

The scalar baseline is broadcast. This is appropriate for a constant null
prediction such as a training-set prevalence. If both models produce
observation-level predictions, pass two equal-length vectors instead.

The following names are equivalent in probability mode:

```python
from imv import calculate_imv, imv_from_probs

assert score == calculate_imv(baseline, enhanced, outcomes)
assert score == imv_from_probs(baseline, enhanced, outcomes)
```

## Likelihood mode

If only the geometric mean likelihood from `ll` has been retained, use the
explicit likelihood function:

```python
from imv import imv_from_likelihoods, ll

a0 = ll(outcomes, baseline)
a1 = ll(outcomes, enhanced)
same_score = imv_from_likelihoods(a0, a1)

assert score == same_score
```

`vanilla_imv(a0, a1)` and `calculate_imv(a0, a1)` also select likelihood mode.
Both inputs must then be scalar values in `(0, 1]`. A two-argument call is never
interpreted as two prediction vectors.

## Inspect the transformation

```python
from imv import get_w

w0 = get_w(a0)
w1 = get_w(a1)

print({
    "baseline_likelihood": a0,
    "enhanced_likelihood": a1,
    "baseline_weight": w0,
    "enhanced_weight": w1,
    "imv": (w1 - w0) / w0,
})
```

Retaining these intermediate values makes an analysis easier to audit and helps
identify below-chance or calibration failures.

## Below-chance diagnostics

```python
import warnings

from imv import BelowChanceLikelihoodWarning, information_deficit

bad_likelihood = 0.1
print(information_deficit(bad_likelihood))

with warnings.catch_warnings():
    warnings.simplefilter("error", BelowChanceLikelihoodWarning)
    get_w(bad_likelihood)  # raises the warning as an exception in strict workflows
```

Do not replace an undefined weight with zero. `information_deficit` remains
ordered below the floor and should be reported separately when no IMV exists.

## Numerical backend

The default `method="brentq"` is deterministic bracketed root finding and should
be used for new work. For numerical parity with pre-1.2.0 analyses:

```python
legacy = calculate_imv(
    baseline,
    enhanced,
    outcomes,
    method="lbfgsb",
)
```

`tolerance` applies only to the legacy optimizer. The low-level `get_w` function
also accepts the historical upper bound `bounds=[(0.5, 0.999)]`; high-level IMV
functions intentionally use the current full representable range.

## Validation rules

- Outcomes are finite binary values `0` and `1`.
- Probabilities are finite values in `[0, 1]`.
- Vectors are non-empty and one-dimensional.
- Both prediction vectors match the outcome length.
- Only genuine numeric scalars broadcast; length-one vectors do not.
- Likelihood-mode values are finite scalar values in `(0, 1]`.

Violations raise `TypeError` or `ValueError` rather than being coerced into an
ambiguous comparison.

