# Compatibility and Migration

## Recommended imports

New code should import public metric functions and evaluator classes from the
package root, and plotting helpers from `imv.utils`:

```python
from imv import AblationIMV, BinaryIMV, MulticlassIMV, vanilla_imv
from imv.utils import plot_imv_heatmap, save_figure
```

## Legacy modules and aliases

The following imports remain valid for legacy and pre-2.0 compatibility:

| Legacy import | Canonical object |
|---|---|
| `from imv.binary import BinaryIMV` | `imv.BinaryIMV` |
| `from imv.binary import IMVEvaluator` | Identity alias of `imv.BinaryIMV` |
| `from imv.multiclass import MulticlassIMV` | `imv.MulticlassIMV` |
| `from imv.multiclass import MultinomialIMV` | Identity alias of `imv.MulticlassIMV` |
| `from imv.ablation import AblationIMV` | `imv.AblationIMV` |
| `from imv.core import calculate_imv` | `imv.calculate_imv` |
| `AblationIMV.ll`, `.get_w`, `.calculate_imv` | Static aliases of core functions |
| `MulticlassIMV.ll`, `.get_w` | Static aliases of core functions |

`imv.core.minimize_me` is retained as the objective used by the legacy optimizer
but is not a recommended public workflow.

## Historical names

`MulticlassIMV.multinominal_imv_matrix` contains a longstanding spelling error.
It is the implemented low-level pairwise method and is retained to avoid breaking
existing code. The plotting names `multinomial_IMV_heatmap` and
`multinomial_IMV_boxplot` use the corrected word but retain historical casing.

## Core call migration

Three equivalent names serve different readability needs:

```python
vanilla_imv(baseline, enhanced, outcomes)
calculate_imv(baseline, enhanced, outcomes)
imv_from_probs(baseline, enhanced, outcomes)
```

`vanilla_imv(a0, a1)`, `calculate_imv(a0, a1)`, and
`imv_from_likelihoods(a0, a1)` accept two already-aggregated scalar likelihoods.
This two-argument mode was added without changing the existing three-argument
probability mode.

## Numerical parity

Version 1.2.0 defaults to bracketed `brentq` inversion and an upper weight bound
of `1 - 1e-12`. Older scripts used L-BFGS-B and often capped the weight at
`0.999`. The old cap pins all sufficiently high likelihoods to one value and is
not a property of the published metric.

Use `method="lbfgsb"` to reproduce the legacy optimizer. For direct calls to
`get_w`, also pass `bounds=[(0.5, 0.999)]` when exact historical bound behavior
is required. Record this choice; do not use it silently in new work.

The current `ll` clips probabilities before logarithms. Early implementations
added epsilon inside each logarithm, which could make a perfect predictor score
slightly above one. Differences are normally on the order of epsilon but are
intentional correctness changes.

## Version support

The package declares Python 3.9 through 3.12 support. Compatibility aliases are
preserved for the current major version; a future removal should be announced in
release notes and accompanied by a deprecation period.
