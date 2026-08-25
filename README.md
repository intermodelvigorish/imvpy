# IMV: InterModel Vigorish

<div align="center">

*A Python implementation of the InterModel Vigorish for comparing probabilistic predictions*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

## Overview

**IMV (InterModel Vigorish)** is a model-agnostic metric for quantifying the change in predictive accuracy between a baseline and an enhanced probabilistic prediction system. It translates each system’s mean log-likelihood into the equivalent weight of a weighted coin, then expresses the enhanced system’s advantage as the house edge—or “vigorish”—of a fair bet priced from the baseline system.

The canonical method is defined for binary outcomes and probability predictions. IMV is always relative to a baseline (which may be prevalence alone), is directional, and is intended to be evaluated on held-out or out-of-sample predictions. It is not mutual information, entropy, accuracy, or a calibrated probability.

### Key Features

**Canonical metric and three extensions:**
- **Vanilla IMV**: Direct comparison from outcomes and probabilities, or from
  two geometric mean likelihoods
- **Binary IMV**: Binary classification with exact IMV-Shapley feature attribution
- **Multi-class IMV**: An extension using one-vs-rest and pairwise comparisons
- **Ablation IMV**: Deep learning ablation studies with GPU support

**Performance:**
- Parallel processing for traditional ML
- GPU acceleration for deep learning (CUDA + Apple Silicon MPS)
- Automatic device detection

**Visualizations:**
- Confusion matrix heatmaps
- Performance comparison plots
- IMV distribution analysis

**Research-oriented:**
- Reproducible results with seed management
- K-fold cross-validation
- Statistical stability across runs

---

## Installation

Requires Python 3.9 or newer.

### Basic installation

```bash
git clone https://github.com/intermodelvigorish/imv_ml_package.git
cd imv_ml_package
pip install .
```

This installs the `imv` package and its dependencies. Verify with:

```bash
python -c "import imv; print(imv.__version__)"
```

> The sources live under `src/`, so `import imv` only works after installing.
> `pyproject.toml` remains the single source of dependency truth;
> `requirements.txt` is a thin convenience entry point that installs the local
> package with every dependency needed by all ten example notebooks.

To install the complete notebook runtime from the repository root:

```bash
pip install -r requirements.txt
```

### Optional extras

All extras are declared in `pyproject.toml` under
`[project.optional-dependencies]`:

| Extra | Installs | For |
|---|---|---|
| `progress` | `tqdm-joblib` | nicer progress bars during coalition fitting |
| `deep-learning` | PyTorch | `AblationIMV` construction, seeding, training, and BERT layer reduction |
| `notebooks` | Jupyter, nbclient, nbformat | running any notebook |
| `examples` | `notebooks` + `ucimlrepo`, XGBoost, LightGBM | the seven tabular examples |
| `examples-deep-learning` | `examples` + `deep-learning` + transformers, datasets | the IMDb, MNIST and HAR ablation examples |
| `test` | pytest, pytest-cov, pyyaml, nbformat | the test suite |
| `docs` | MkDocs Material + mkdocstrings | building or serving the documentation |
| `dev` | `progress` + `notebooks` + `test` + `docs` + build, ruff, mypy | developing the library |

```bash
pip install ".[examples]"    # e.g. to run the tabular notebooks
pip install -e ".[dev]"      # e.g. to work on the library
```

`AblationIMV` selects CUDA, then Apple Silicon MPS, then CPU automatically; no
platform-specific configuration is needed. Static matrix calculation and
averaging from saved predictions need no PyTorch or evaluator instance.

### Development installation

```bash
pip install -e ".[dev]"
pytest                        # fast suite (the merge gate)
pytest -m ""                  # include slow and deep-learning tests
ruff check .                  # lint, configured in pyproject.toml
```

### Conda

```bash
conda env create -f environment.yml
conda activate imv
```

---

## Quick Start

### 1. Vanilla IMV

The original PLOS supplementary example uses a scalar baseline probability and
an observation-level enhanced prediction. Scalars are broadcast; NumPy arrays,
pandas `Series`, lists and tuples are accepted positionally:

```python
import numpy as np
from imv import ll, vanilla_imv

observed = np.array([
    0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
])
enhanced = np.repeat([0.5, 0.9], 20)

vanilla_imv(0.55, enhanced, observed)
# 0.23722913125143966

# The same Eq. 6 calculation from already-aggregated Eq. 2 likelihoods:
vanilla_imv(ll(observed, 0.55), ll(observed, enhanced))
# 0.23722913125143966
```

The argument order is always baseline, enhanced, then observed outcomes. In the
two-argument form, both values must instead be scalar geometric mean
likelihoods. Use `imv_from_likelihoods(a0, a1)` when you want to make that mode
explicit. Use held-out probabilities for model comparison; in-sample values
favor the more complex model.

### 2. Binary IMV: Binary Classification

Compute IMV-based feature importance for binary classification:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imv import BinaryIMV

# Load data
data = pd.read_csv('adult_income.csv')

# Create evaluator
evaluator = BinaryIMV(
    data=data,
    outcome_variable='income_>50K',
    optional_explanatory_variables=['age', 'education_years', 'hours_per_week'],
    model_creator=lambda: LogisticRegression(max_iter=1000),
    split_method='kfold',
    n_splits=5,
    prop_test=0.2,
    model_type='classification',
    random_seed=42,
    verbose=True,          # progress bar and per-variable values; default False
)

# Trains a model per feature subset per fold: 2**n_features * n_splits * 2 fits
evaluator.run_evaluation()

# Per-feature attribution, plus a plot
for feature in ['age', 'education_years', 'hours_per_week']:
    print(feature, evaluator.calculate_imvshapley_value(feature))

fig, ax = evaluator.evaluate_imvshapley(figsize=(12, 4))
```

**Output** (values depend on your data; `verbose=True` is required for the first line):
```
Best explanatory variables' combination: ('age', 'education_years'), IMV: 0.234
age 0.145
education_years 0.112
hours_per_week 0.068
```

The returned values are rounded to 3 decimals. Before rounding, they sum to the
full-coalition value minus the empty-coalition value; rounding can leave a small
residual. A negative value means the feature *reduced* out-of-fold information.

### 3. Multi-class IMV: Multi-class Classification

Analyze multi-class classification problems:

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imv import MulticlassIMV

# Load multi-class data
data = pd.read_csv('nursery.csv')

# Create evaluator
evaluator = MulticlassIMV(
    data=data,
    outcome_variable='outcome',
    model_creator=lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    n_splits=10,
    optional_explanatory_variables=['parents', 'has_nurs', 'form', 'children'],
    random_state=42
)

# Calculate IMV confusion matrix
_, imv_matrix = evaluator.k_fold_imv_matrix()
print(imv_matrix)

# Visualize
fig, ax = evaluator.multinomial_IMV_heatmap(imv_matrix, figsize=(6, 6))
```

**Interpretation:**
- Diagonal values are always 0 (a class compared to itself)
- Off-diagonal values show information gain when separating one class from another
- Higher values indicate better class separation
- This matrix is **symmetric by construction**: `ll` is invariant under
  `(y, p) -> (1-y, 1-p)`, and pairwise renormalisation gives `p_j = 1 - p_i`.
  Do not read it like the directional ablation matrix below.

### 4. Ablation IMV: Deep Learning

Quantify the importance of model components in deep learning:

```python
import torch
from transformers import DistilBertForSequenceClassification
from imv import AblationIMV

# Initialize (automatically detects GPU)
ablator = AblationIMV(random_seed=42)
# Output: Using device: Apple Silicon GPU (MPS)

def fresh_model():
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

# reduce_bert_layers modifies the model IN PLACE, so build each variant from a
# freshly loaded checkpoint. Reusing one object would leave every variant
# pointing at the same, most-truncated model.
variants = {
    '6-layer': fresh_model(),
    '4-layer': ablator.reduce_bert_layers(fresh_model(), num_layers_to_keep=4),
    '2-layer': ablator.reduce_bert_layers(fresh_model(), num_layers_to_keep=2),
}

# Train each variant and collect its held-out predictions
results = {}
for name, model in variants.items():
    result = ablator.train_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        num_epochs=3,
        lr=2e-5,
        seed=42
    )
    results[name] = result['test_predictions']

# Compute the directional IMV matrix (rows = enhanced, columns = basic)
imv_matrix = ablator.calculate_imv_matrix(results)
print(imv_matrix)
```

Unlike the multiclass pairwise matrix, this one **is** directional: each cell
uses a different baseline in the denominator, so values are not comparable
across columns.

---

## Project Structure

```
imv_ml_package/
├── src/imv/                    # Installable package (src layout)
│   ├── utils/                  # Shared metric and plotting utilities
│   ├── shap_imv/               # Binary exact SHAP-IMV
│   ├── multi_imv/              # Multiclass IMV
│   └── ablation_imv/           # Ablation training and comparison
├── examples/                   # Ten executed notebooks, each self-contained
│   ├── shap_imv/               # Executed binary SHAP-IMV notebooks
│   ├── multi_imv/              # Executed multiclass IMV notebooks
│   └── ablation_imv/           # Executed model-ablation notebooks
├── documentation/              # Complete MkDocs guides and API reference
├── config/settings.yaml        # Documented defaults and reproducibility profiles
├── tests/                      # Unit, integration and contract tests
├── mkdocs.yml                  # Strict documentation build and navigation
├── pyproject.toml              # Package metadata, dependencies, lint and test config
├── requirements.txt            # One-command runtime for all example notebooks
├── environment.yml             # Conda environment
└── README.md                   # This file
```

No dataset is stored in this repository. Every notebook downloads what it needs
into a cache outside the working tree, so `examples/` runs from a cold clone.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Documentation site](documentation/index.md)** | Installation, concepts, task guides, API reference and research guidance |
| **[Examples](examples/README.md)** | The ten executed notebooks, their datasets and their runtimes |
| **[Settings reference](config/settings.yaml)** | Machine-readable defaults and parity/production profiles |
| **[API reference](documentation/api/core.md)** | Generated from every public function and evaluator docstring |

Build or serve the complete documentation from the repository root:

```bash
pip install -e ".[docs]"
mkdocs serve               # local site at http://127.0.0.1:8000
mkdocs build --strict      # production build under site/
```

### Scientific notes worth knowing before you interpret output

- **IMV is a bounded likelihood transformation**, not mutual information, entropy
  or a calibrated probability. Magnitudes depend on the baseline model, so values
  computed against different baselines are not comparable.
- **The metric is directional.** Reversing basic and enhanced changes the
  denominator, so `IMV(B→E)` and `IMV(E→B)` are not negatives of each other. The
  ablation matrix is therefore neither symmetric nor antisymmetric. The
  multiclass **pairwise** matrix is a different construction and *is* symmetric:
  there the two cells swap labels rather than swapping basic and enhanced.
- **Likelihoods below 0.5 have no exact equivalent coin.** `get_w` returns the
  0.5 boundary for a small documented finite-sample residual; further below it
  returns `NaN` and warns. `information_deficit(a)` reports `log(2a)` nats
  throughout. A below-chance likelihood is evidence of miscalibration, since
  every calibrated predictor scores at least 0.5 in expectation.
- **SHAP-IMV is not ordinary SHAP.** It is a global attribution of held-out
  information, not a local prediction attribution. A negative value means the
  feature *reduced* out-of-fold information — it does **not** mean the feature
  indicates the negative class.
- **Fold-to-fold spread is not a confidence interval.** Folds share training rows.
- Report the split method, seed, baseline model, estimator, calibration procedure
  and package version with any IMV result.

### Citation

This package is based on:

Domingue BW, Rahal C, Faul J, Freese J, Kanopka K, Rigos A, et al. (2025).
“The InterModel Vigorish (IMV) as a flexible and portable approach for
quantifying predictive accuracy with binary outcomes.” *PLOS ONE*, 20(3),
e0316491. [https://doi.org/10.1371/journal.pone.0316491](https://doi.org/10.1371/journal.pone.0316491)

---

## Example analyses

Ten executed notebooks live under `examples/`. Every one runs from a cold
clone: nothing is read from the repository, and each notebook downloads what it
needs into a cache outside the working tree. Every tabular estimator and every
ablation architecture is run under ten seeds (42–51); the tabular examples
compare logistic regression, XGBoost and LightGBM, while the ablation examples
compare deep-learning architectures.

The notebooks import this checkout's package directly. Each has an executable
provenance guard that verifies `imv.__file__` resolves to `src/imv`, and the
repository runtime installs the project in editable mode via `requirements.txt`.

These are **our own runs, not replications of the published figures.** Of the
datasets in the published work, only Adult Income, Nursery and IMDb have
generating code in the original research notebooks.

### Binary — exact SHAP-IMV (`examples/shap_imv/`)

| Notebook | Outcome | Features | Source |
|---|---|---|---|
| `shap_imv_adult_income.ipynb` | income > 50K | 6 demographic / human-capital | UCI id 2 |
| `shap_imv_titanic.ipynb` | survived | 6 incl. `Title` from the name field | OpenML `titanic` v1 |
| `shap_imv_breast_cancer.ipynb` | malignant | 6 least-collinear "mean" measurements | UCI id 17 |
| `shap_imv_wine_quality.ipynb` | `quality >= 6` | 6 chemical properties | UCI id 186 |

Feature counts are capped deliberately: exact SHAP-IMV costs
`2**n_features * n_splits * 2` fits per seed per estimator.

### Multiclass (`examples/multi_imv/`)

| Notebook | Classes | Features | Source |
|---|---|---|---|
| `multi_imv_nursery.ipynb` | 3 of 5 (two negligible classes dropped) | 8 categorical | UCI id 76 |
| `multi_imv_car_evaluation.ipynb` | 4 | 6 categorical | UCI id 19 |
| `multi_imv_dry_bean.ipynb` | 7 | 16 continuous | UCI id 602 |

### Ablation (`examples/ablation_imv/`)

| Notebook | What it ablates | Source |
|---|---|---|
| `ablation_imv_imdb.ipynb` | DistilBERT layers, attention, FFN, layer norm | HF `stanfordnlp/imdb` |
| `ablation_imv_mnist.ipynb` | CNN convolution, hidden layer, dropout, complete feature extractor | OpenML `mnist_784` v1 |
| `ablation_imv_har.ipynb` | bidirectionality, recurrent depth, temporal attention, temporal order | UCI HAR id 240 |

All three notebooks run 50 restartable fits: any variant whose predictions
already exist on disk and pass its alignment checks is reused. IMDb is roughly
two hours on the accelerated reference machine and can take several hours on
CPU alone; MNIST uses the canonical 60,000/10,000 split and one epoch per fit;
HAR preserves UCI's subject-disjoint split and uses 15 epochs per fit.

Figures and result tables are embedded in the executed notebooks. Downloaded
datasets, restart predictions, and optional file exports stay outside the
repository under `~/.cache/imv`. See [examples/README.md](examples/README.md) for
runtimes and conventions.

Every notebook figure is also exported to the external artifact cache in three
publication-ready forms: PNG at 800 DPI, PDF, and SVG. The shared
`imv.utils.save_figure` helper enforces this consistently.

### Running Examples

```bash
pip install -r requirements.txt             # all ten notebooks
jupyter lab examples/
```

The notebooks may be launched from their own directories without modifying the
checkout; all caches and optional exports are written under `~/.cache/imv`.

---

## Testing

```bash
pytest                                          # fast suite; the merge gate
pytest --cov=src/imv --cov-report=term-missing  # with coverage
pytest -m slow -o addopts='-ra'                 # research-scale demonstrations
pytest -m deep_learning -o addopts='-ra'        # needs the deep-learning extra
pytest -m ""                                    # everything
```

The default marker expression in `pyproject.toml` excludes tests marked `slow`,
`network` and `deep_learning`, so a bare `pytest` is fast and needs no network.
Alongside the unit tests, `tests/test_settings_contract.py` holds
`config/settings.yaml` to the live Python signatures, and
`tests/test_repository_contract.py` checks that every example still ships,
downloads its own data and uses the required ten seeds.

---

## API Reference

Signatures below are the live defaults. `config/settings.yaml` carries the same
values in machine-readable form and is held to the Python signatures by
`tests/test_settings_contract.py`.

### Core functions

```python
from imv import (
    calculate_imv,
    get_w,
    imv_from_likelihoods,
    information_deficit,
    ll,
    vanilla_imv,
)

ll(x, p, epsilon=1e-9)                       # geometric mean Bernoulli likelihood
get_w(a, guess=0.5, bounds=[(0.5, 1 - 1e-12)], tolerance=1e-9,
      chance_tolerance_nats=0.5, method="brentq")
calculate_imv(y_basic, y_enhanced, y=None, epsilon=1e-9, tolerance=1e-9,
              method="brentq")
vanilla_imv(baseline, enhanced, outcomes=None, epsilon=1e-9, tolerance=1e-9,
            method="brentq")
imv_from_likelihoods(likelihood_basic, likelihood_enhanced,
                     tolerance=1e-9, method="brentq")
information_deficit(a)                       # log(2a) nats; defined below chance
```

In three-argument mode, `calculate_imv` and `vanilla_imv` accept one-dimensional
NumPy arrays, pandas `Series`, lists or tuples. Python and NumPy numeric scalars
are valid constant predictions and are broadcast to the outcome length; scalar
outcomes represent a one-observation dataset. In two-argument mode, both inputs
are scalar geometric mean likelihoods in `(0, 1]`. A length-one vector is never
broadcast implicitly.

`method="brentq"` brackets the root of `g(w) - log(a)` and cannot stall.
`method="lbfgsb"` reproduces pre-1.2.0 published numbers; pair it with
`bounds=[(0.5, 0.999)]` for an exact legacy match. `guess` and `tolerance` apply
to `"lbfgsb"` only.

### BinaryIMV Class

```python
from imv import BinaryIMV

evaluator = BinaryIMV(
    data: pd.DataFrame,
    outcome_variable: str,
    optional_explanatory_variables: list[str],
    model_creator: Callable,          # returns a fresh fit/predict_proba estimator
    split_method: str = 'kfold',      # also: stratified_kfold,
                                      # train_test_split, stratified_train_test_split
    n_splits: int = 5,
    prop_test: float = 0.2,           # holdout modes only
    model_type: str = 'classification',
    all_combinations_imv: dict | None = None,
    random_seed: int = 42,
    n_jobs: int = 1,                  # joblib workers across coalitions
    verbose: bool = False,
)
```

**Key Methods:**
- `run_evaluation()`: fit every feature coalition; stores and returns the mapping
- `calculate_imvshapley_value(variable)`: exact SHAP-IMV for one feature
- `evaluate_imvshapley(ax=None, figsize=(12, 4))`: all values, plus a bar plot
- `plot_single_var_combinations_layered_violin_centralized_zero(ax=None, figsize=(6, 4))`

Use `split_method='kfold'` or `'train_test_split'` for parity with the original
notebooks, and the stratified variants for a new analysis — then report which.

### MulticlassIMV Class

```python
from imv import MulticlassIMV

evaluator = MulticlassIMV(
    data: pd.DataFrame,
    outcome_variable: str,
    model_creator: Callable,
    n_splits: int = 10,
    optional_explanatory_variables: list[str] | None = None,
    random_state: int | None = None,
    stratified: bool = False,         # False preserves the original KFold
    verbose: bool = False,
)
```

**Key Methods:**
- `k_fold_one_vs_all()`: per-fold class-vs-rest values and their mean
- `k_fold_imv_matrix()`: per-fold pairwise matrices and their mean
- `one_vs_all_single_fold(data, outcome_variable, p_base, p_enhanced, classes=None)`
- `multinominal_imv_matrix(...)`: the historical misspelling; low-level pairwise
- `multinomial_IMV_heatmap(imv_matrix, ax=None, figsize=(6, 6))`
- `multinomial_IMV_boxplot(imv_results, figsize=(6, 6), ax=None)`

Both numeric and string labels are supported. Pass `classes=model.classes_` to
the low-level per-fold methods whenever a fold might not hold every trained
class; the `k_fold_*` methods pass it automatically.

### AblationIMV Class

```python
from imv import AblationIMV

ablator = AblationIMV(random_seed: int = 42)   # CUDA > MPS > CPU, auto-detected
```

**Key Methods:**
- `train_and_evaluate(model, train_dataloader, test_dataloader, num_epochs=3,
  lr=2e-5, optimizer_class=None, scheduler_fn=None, max_grad_norm=None,
  seed=None, verbose=True)`
- `calculate_imv_matrix(predictions_dict, target_column='True Label',
  prob_column='Positive Probability')`: directional matrix, rows = enhanced
- `average_imv_matrices(matrices_list)`: mean across seeds or runs
- `reduce_bert_layers(model, num_layers_to_keep)`: truncates **in place**

`calculate_imv_matrix`, `average_imv_matrices` and `reduce_bert_layers` are
static. Matrix calculation and averaging work without PyTorch or an
`AblationIMV` instance; construction, training, seeding and BERT layer reduction
need the `deep-learning` extra.

---

## Performance

Exact SHAP-IMV enumerates the full power set, so cost is
`2**n_features * n_splits * 2` fits per estimator per seed. Roughly:

| Configuration | Coalitions | Model fits | Feasibility |
|---|---:|---:|---|
| 5 features, 5-fold | 32 | 320 | seconds |
| 10 features, 10-fold | 1,024 | 20,480 | minutes |
| 11 features, 10-fold | 2,048 | 40,960 | tens of minutes |
| 15 features, 10-fold | 32,768 | 655,360 | impractical for most estimators |

`n_jobs` defaults to **1**, so the library does not silently occupy every core.
Set it explicitly to parallelise across coalitions, and avoid nesting it with
estimator-level parallelism (`n_jobs` inside XGBoost or LightGBM). Exact mode is
practical up to roughly 10–15 inexpensive features; beyond that the method needs
sampled coalitions rather than a bigger machine.

---

## Hardware Requirements

### Minimum Requirements
- **CPU:** Dual-core processor
- **RAM:** 4GB
- **Python:** 3.9+

### Recommended for Deep Learning
- **GPU:**
  - NVIDIA GPU with 8GB+ VRAM (CUDA)
  - Apple M1/M2/M3 (8GB+ unified memory)
- **RAM:** 16GB+
- **Storage:** 10GB+ for model checkpoints

---

## Troubleshooting

### Slow performance
- Reduce the number of features — cost is exponential in that count, and linear
  in everything else
- Reduce `n_splits`
- Use a holdout `split_method` instead of `'kfold'`
- Raise `n_jobs` (it defaults to 1)

### Memory issues
- Test with fewer features first
- Subsample rows, or cut `n_splits`; note that cost is driven by the `2**n_features`
  coalition count, so dropping a feature halves it

### `ImportError: No module named imv`
The sources live under `src/`, so the package must be installed — a dependency
list alone is not enough:

```bash
pip install -e ".[dev]"
```

### `BelowChanceLikelihoodWarning` and `NaN` results
A geometric mean likelihood below 0.5 has no exact equivalent coin weight.
`get_w` uses the boundary only within its documented finite-sample tolerance;
further below it returns `NaN` and warns. This diagnoses miscalibration rather
than weak discrimination: recalibrate on data separate from the scored holdout,
and use `information_deficit(a)` to report the shortfall.

### `IncompleteCoalitionWarning`
`calculate_imvshapley_value` needs all `2**n_features` coalitions. Missing ones
are treated as `IMV = 0`, which breaks additivity, so the result is not a valid
Shapley value. Re-run `run_evaluation()` over the full power set.

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/intermodelvigorish/imv_ml_package/issues)

---

<div align="center">

[Back to Top](#imv-intermodel-vigorish)

</div>
