# IMVpy

<div align="center">

*InterModel Vigorish for probabilistic model comparison and attribution*

[![PyPI](https://img.shields.io/pypi/v/imvpy.svg)](https://pypi.org/project/imvpy/)
[![Python](https://img.shields.io/pypi/pyversions/imvpy.svg)](https://pypi.org/project/imvpy/)
[![CI](https://github.com/intermodelvigorish/imvpy/actions/workflows/ci.yml/badge.svg)](https://github.com/intermodelvigorish/imvpy/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-IMVpy-0f766e.svg)](https://intermodelvigorish.github.io/imvpy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/intermodelvigorish/imvpy/blob/main/LICENSE)

</div>

IMVpy implements the **InterModel Vigorish (IMV)**, a model-agnostic metric
for comparing two systems that produce probabilities for a binary outcome. It
maps each system's geometric mean Bernoulli likelihood to an equivalent
weighted coin and reports the enhanced system's advantage relative to the
baseline system.

The project, PyPI distribution, and Python import are all named **IMVpy** /
**`imvpy`**:

```bash
python -m pip install imvpy
```

```python
import imvpy

print(imvpy.__version__)
```

IMVpy deliberately installs no `imv` namespace alias, avoiding ambiguity with
unrelated distributions.

## Workflows

| Workflow | Entry point | Purpose |
|---|---|---|
| Vanilla IMV | `vanilla_imv` | Compare binary probability predictions or aggregated likelihoods |
| Exact SHAP-IMV | `BinaryIMV` | Attribute global held-out IMV across the complete feature power set |
| Multiclass IMV | `MulticlassIMV` | Compute one-vs-rest and pairwise multiclass extensions |
| Model ablation | `AblationIMV` | Compare aligned predictions from model variants directionally |

All scoring uses probabilities rather than hard labels. IMV is relative to a
declared baseline, directional, and intended for held-out or out-of-sample
predictions.

## Installation

IMVpy supports Python 3.9 and newer.

```bash
python -m pip install imvpy
```

Optional features are installed explicitly:

```bash
python -m pip install "imvpy[progress]"       # joblib-aware progress bars
python -m pip install "imvpy[deep-learning]" # PyTorch training helpers
```

For development from a clone:

```bash
git clone https://github.com/intermodelvigorish/imvpy.git
cd imvpy
python -m pip install -e ".[dev]"
```

## Quick Start

### Vanilla IMV

Pass baseline probabilities, enhanced probabilities, and observed binary
outcomes. Numeric scalars are broadcast, while NumPy arrays, pandas `Series`,
lists, and tuples are accepted as observation-level inputs.

```python
from imvpy import ll, vanilla_imv

outcomes = [1, 0, 1, 0]
enhanced = [0.85, 0.15, 0.75, 0.25]

score = vanilla_imv(0.5, enhanced, outcomes)
print(score)

# The equivalent calculation from already-aggregated likelihoods:
same_score = vanilla_imv(ll(outcomes, 0.5), ll(outcomes, enhanced))
```

The argument order is always baseline, enhanced, then outcomes. A two-argument
call instead treats both inputs as scalar geometric mean likelihoods. Use
`imv_from_likelihoods` when that mode should be explicit.

### Exact SHAP-IMV

`BinaryIMV` fits every feature coalition and calculates exact global Shapley
attributions of held-out IMV.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from imvpy import BinaryIMV

features, outcome = make_classification(
    n_samples=300,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)
columns = ["x1", "x2", "x3"]
data = pd.DataFrame(features, columns=columns).assign(outcome=outcome)

evaluator = BinaryIMV(
    data=data,
    outcome_variable="outcome",
    optional_explanatory_variables=columns,
    model_creator=lambda: LogisticRegression(max_iter=2000),
    split_method="stratified_kfold",
    n_splits=3,
    random_seed=42,
)
evaluator.run_evaluation()

values = {
    feature: evaluator.calculate_imvshapley_value(feature)
    for feature in columns
}
print(values)
```

Exact SHAP-IMV costs `2**n_features * n_splits * 2` model fits. Keep the
feature universe small enough to evaluate the complete power set.

### Multiclass IMV

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imvpy import MulticlassIMV

data = load_iris(as_frame=True).frame.rename(columns={"target": "species"})
feature_columns = [column for column in data.columns if column != "species"]

evaluator = MulticlassIMV(
    data=data,
    outcome_variable="species",
    optional_explanatory_variables=feature_columns,
    model_creator=lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    ),
    n_splits=5,
    random_state=42,
    stratified=True,
)

fold_matrices, mean_matrix = evaluator.k_fold_imv_matrix()
print(mean_matrix)
```

The pairwise multiclass matrix is symmetric by construction. This differs from
the directional model-ablation matrix.

### Model Ablation

Matrix calculation is framework-independent and does not require PyTorch. Each
variant must contain probabilities for the same held-out rows in the same
order.

```python
import pandas as pd

from imvpy import AblationIMV

predictions = {
    "baseline": pd.DataFrame(
        {
            "True Label": [1, 0, 1, 0],
            "Positive Probability": [0.70, 0.30, 0.65, 0.35],
        }
    ),
    "enhanced": pd.DataFrame(
        {
            "True Label": [1, 0, 1, 0],
            "Positive Probability": [0.90, 0.10, 0.80, 0.20],
        }
    ),
}

matrix = AblationIMV.calculate_imv_matrix(predictions)
print(matrix)
```

Install `imvpy[deep-learning]` to construct `AblationIMV` and use its PyTorch
seeding and training helpers. Static matrix calculation and averaging remain
available from the base installation.

## Plotting and Export

Plotting functions accept an existing Matplotlib axis or create one. The shared
export helper writes PNG at 800 DPI plus PDF and SVG:

```python
from imvpy.utils import save_figure

figure, axis = evaluator.multinomial_IMV_heatmap(mean_matrix)
paths = save_figure(figure, "artifacts/multiclass_imv")
```

## Interpretation

- IMV is a bounded likelihood transformation, not mutual information,
  accuracy, entropy, or a calibrated probability.
- IMV is directional. Reversing baseline and enhanced predictions does not
  generally negate the original value.
- Values calculated against different baselines are not directly comparable.
- A geometric mean likelihood sufficiently below 0.5 has no equivalent-coin
  root. IMVpy warns and returns `NaN` rather than silently substituting a value.
- SHAP-IMV is a global attribution of held-out predictive information, not a
  local explanation of one prediction.
- Fold or seed variation is not a confidence interval without a valid
  inferential procedure.

Report the package version, baseline, estimator, split design, seed,
calibration procedure, and any below-chance warnings with each result.

## Documentation

- [Documentation](https://intermodelvigorish.github.io/imvpy/)
- [Installation guide](https://intermodelvigorish.github.io/imvpy/getting-started/installation/)
- [API reference](https://intermodelvigorish.github.io/imvpy/api/core/)
- [Changelog](https://github.com/intermodelvigorish/imvpy/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/intermodelvigorish/imvpy/blob/main/CONTRIBUTING.md)

## Citation

IMVpy implements the method introduced in:

> Domingue BW, Rahal C, Faul J, Freese J, Kanopka K, Rigos A, et al. (2025).
> "The InterModel Vigorish (IMV) as a flexible and portable approach for
> quantifying predictive accuracy with binary outcomes." *PLOS ONE*, 20(3),
> e0316491. [doi:10.1371/journal.pone.0316491](https://doi.org/10.1371/journal.pone.0316491)

Use the repository's
[citation metadata](https://github.com/intermodelvigorish/imvpy/blob/main/CITATION.cff)
and record the exact package version when reporting software-derived results.

## License

IMVpy is distributed under the
[MIT License](https://github.com/intermodelvigorish/imvpy/blob/main/LICENSE).
