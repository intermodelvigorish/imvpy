# IMV

`imv` implements InterModel Vigorish for binary probabilistic predictions and
three model-comparison workflows built on the same metric:

| Workflow | Question answered | Entry point |
|---|---|---|
| Vanilla IMV | How much transformed predictive information does one binary predictor add over another? | `vanilla_imv` |
| Exact SHAP-IMV | How is a model's global held-out IMV distributed over feature coalitions? | `BinaryIMV` |
| Multiclass IMV | How much do features improve one-vs-rest or pairwise class discrimination? | `MulticlassIMV` |
| Ablation IMV | How much predictive information changes between aligned architecture variants? | `AblationIMV` |

The package consumes probabilities, not hard labels. Its canonical calculation is
model-agnostic and accepts NumPy arrays, pandas Series, Python sequences, and
numeric scalars where a constant prediction is meaningful.

```python
from imv import vanilla_imv

score = vanilla_imv(
    baseline=0.5,
    enhanced=[0.9, 0.1, 0.8, 0.2],
    outcomes=[1, 0, 1, 0],
)
print(score)
```

## Documentation map

- Start with [Installation](getting-started/installation.md), then run the
  [first IMV calculation](getting-started/first-imv.md).
- Read the [metric definition](concepts/metric.md) before interpreting or
  reporting values.
- Select a task-specific guide for [vanilla IMV](guides/vanilla.md),
  [SHAP-IMV](guides/shap.md), [multiclass IMV](guides/multiclass.md), or
  [model ablation](guides/ablation.md).
- Use the [API reference](api/core.md) for signatures generated from the source.

## Scope and guarantees

The implemented metric is the binary-outcome construction in Domingue, Rahal,
et al. It is a bounded transformation of geometric mean Bernoulli likelihoods.
It is not mutual information, a probability, or an estimator for regression.

All high-level evaluators score held-out predictions. The library validates
binary outcomes, probability ranges, aligned ablation labels, and multiclass
probability-column order where it has enough information to do so. It does not
automatically calibrate models, choose a scientifically valid split strategy, or
turn repeated seeds or folds into confidence intervals; those remain analysis
decisions.

## Package version

This documentation describes `imv` 1.2.0. The package version, documented
defaults, and Python signatures are checked against one another in the test
suite.
