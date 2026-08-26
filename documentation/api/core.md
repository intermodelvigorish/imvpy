# Core API

Import the supported metric surface from `imvpy`:

```python
from imvpy import (
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

::: imvpy.utils.core.vanilla_imv

::: imvpy.utils.core.calculate_imv

::: imvpy.utils.core.imv_from_probs

::: imvpy.utils.core.imv_from_likelihoods

## Transformation primitives

::: imvpy.utils.core.ll

::: imvpy.utils.core.get_w

::: imvpy.utils.core.information_deficit

## Warning

::: imvpy.utils.core.BelowChanceLikelihoodWarning
    options:
      members: false

## Legacy optimization objective

`minimize_me` remains importable from `imvpy.core` for old code and for exact
legacy-backend inspection. New code should call `get_w` rather than this
optimizer objective directly.

::: imvpy.utils.core.minimize_me
