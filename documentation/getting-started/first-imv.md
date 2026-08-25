# First IMV

The original paper's supplementary toy calculation has one constant baseline
prediction, observation-level enhanced predictions, and binary outcomes.

```python
import numpy as np

from imv import ll, vanilla_imv

observed = np.array([
    0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
])
enhanced = np.repeat([0.5, 0.9], 20)

score = vanilla_imv(0.55, enhanced, observed)
print(score)
# 0.23722913125143966
```

The positional order is always baseline, enhanced, outcomes. A genuine scalar
prediction is broadcast to the outcome length; a one-element list or array is
not, because silently broadcasting an accidentally truncated vector would hide
an alignment error.

## Equivalent call forms

`vanilla_imv` and `calculate_imv` expose the same two modes.

```python
# Observation-level probabilities plus outcomes
from_probabilities = vanilla_imv(0.55, enhanced, observed)

# Already-aggregated geometric mean likelihoods
from_likelihoods = vanilla_imv(
    ll(observed, 0.55),
    ll(observed, enhanced),
)

assert from_probabilities == from_likelihoods
```

Use the explicit names when call-site clarity matters:

```python
from imv import imv_from_likelihoods, imv_from_probs

imv_from_probs(0.55, enhanced, observed)
imv_from_likelihoods(ll(observed, 0.55), ll(observed, enhanced))
```

## Accepted inputs

Prediction and outcome vectors may be NumPy arrays, pandas Series, lists, or
tuples. Python and NumPy numeric scalars are valid. Inputs must be finite and
one-dimensional; outcomes must contain only 0/1 and probabilities must lie in
`[0, 1]`.

```python
import pandas as pd

y = pd.Series([1, 0, 1, 0])
p0 = 0.5
p1 = pd.Series([0.8, 0.2, 0.7, 0.1])

vanilla_imv(p0, p1, y)
```

In the two-argument form, both values are interpreted as scalar geometric mean
likelihoods in `(0, 1]`. If you intended to pass two probability vectors, supply
the outcome vector as the third argument.

## Before reporting the number

Use predictions from observations not used to fit the compared models. Keep the
same rows and outcomes in the same order for both models. Report the baseline,
split procedure, seed, estimator, calibration procedure, and package version.
See [Statistical Practice](../concepts/statistical-practice.md) for the full
checklist.

