# Core API

Import the supported metric surface from `imv`:

```python
from imv import (
    BelowChanceLikelihoodWarning,
    calculate_imv,
    get_w,
    imv_from_likelihoods,
    imv_from_probs,
    information_deficit,
    ll,
    vanilla_imv,
)
```

`vanilla_imv` is the discoverable canonical name. `calculate_imv` retains the
historical name and supports the same two call forms; `imv_from_probs` and
`imv_from_likelihoods` make the intended input mode explicit.

## Metric entry points

::: imv.utils.core.vanilla_imv

::: imv.utils.core.calculate_imv

::: imv.utils.core.imv_from_probs

::: imv.utils.core.imv_from_likelihoods

## Transformation primitives

::: imv.utils.core.ll

::: imv.utils.core.get_w

::: imv.utils.core.information_deficit

## Warning

::: imv.utils.core.BelowChanceLikelihoodWarning
    options:
      members: false

## Legacy optimization objective

`minimize_me` remains importable from `imv.core` for old code and for exact
legacy-backend inspection. New code should call `get_w` rather than this
optimizer objective directly.

::: imv.utils.core.minimize_me

