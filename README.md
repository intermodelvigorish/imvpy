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

**Three Powerful Modules:**
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
> There is no `requirements.txt`: `pyproject.toml` is the single source of
> dependency truth, and installing the package installs its dependencies.

### Optional extras

All extras are declared in `pyproject.toml` under
`[project.optional-dependencies]`:

| Extra | Installs | For |
|---|---|---|
| `progress` | `tqdm-joblib` | nicer progress bars during coalition fitting |
| `deep-learning` | PyTorch | `AblationIMV` **training** only |
| `notebooks` | Jupyter, nbclient, nbformat | running any notebook |
| `examples` | `notebooks` + `ucimlrepo`, XGBoost, LightGBM | the seven tabular examples |
| `examples-deep-learning` | `examples` + `deep-learning` + transformers, datasets | the IMDb ablation example |
| `test` | pytest, pytest-cov, pyyaml, nbformat | the test suite |
| `dev` | `progress` + `notebooks` + `test` + build, ruff, mypy | developing the library |

```bash
pip install ".[examples]"    # e.g. to run the tabular notebooks
pip install -e ".[dev]"      # e.g. to work on the library
```

`AblationIMV` selects CUDA, then Apple Silicon MPS, then CPU automatically; no
platform-specific configuration is needed. Computing IMV from saved predictions
needs no PyTorch at all — only training does.

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

### 1. Binary IMV: Binary Classification

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

The returned values are rounded to 3 decimals and sum to the full-model IMV.
A negative value means the feature *reduced* out-of-fold information.

### 2. Multi-class IMV: Multi-class Classification

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

### 3. Ablation IMV: Deep Learning

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
├── examples/                   # Eight executed notebooks, each self-contained
│   ├── shap_imv/               # Notebooks, results, figures
│   ├── multi_imv/              # Notebooks, results, figures
│   └── ablation_imv/           # Notebooks, results, figures
├── config/settings.yaml        # Documented defaults and reproducibility profiles
├── tests/                      # Unit, integration and contract tests
├── pyproject.toml              # Package metadata, dependencies, lint and test config
├── environment.yml             # Conda environment
└── README.md                   # This file
```

No dataset is stored in this repository. Every notebook downloads what it needs
into a cache outside the working tree, so `examples/` runs from a cold clone.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Examples](examples/README.md)** | The eight executed notebooks, their datasets and their runtimes |
| **[Settings reference](config/settings.yaml)** | Machine-readable defaults and parity/production profiles |
| Docstrings | Every public function documents its parameters, boundary behaviour and numerical choices |

The changelog and the methodology, API, data-cleaning, parity and audit
write-ups are maintained separately and are not distributed with the package.
Contact the authors if you need them for review.

### Scientific notes worth knowing before you interpret output

- **IMV is a bounded likelihood transformation**, not mutual information, entropy
  or a calibrated probability. Magnitudes depend on the baseline model, so values
  computed against different baselines are not comparable.
- **The metric is directional.** Reversing basic and enhanced changes the
  denominator, so `IMV(B→E)` and `IMV(E→B)` are not negatives of each other. The
  ablation matrix is therefore neither symmetric nor antisymmetric. The
  multiclass **pairwise** matrix is a different construction and *is* symmetric:
  there the two cells swap labels rather than swapping basic and enhanced.
- **Likelihoods below 0.5 have no equivalent coin.** `get_w` returns `NaN` there
  and warns; `information_deficit(a)` reports `log(2a)` nats instead. A
  below-chance likelihood is evidence of miscalibration, since every calibrated
  predictor scores at least 0.5.
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

Eight executed notebooks live under `examples/`. Every one runs from a cold
clone: nothing is read from the repository, and each notebook downloads what it
needs into a cache outside the working tree. All of them use five seeds (42–46)
and compare logistic regression, XGBoost and LightGBM.

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

The IMDb notebook is roughly two hours for 25 fine-tuning runs, and is
restartable: any variant whose predictions already exist on disk is reused.

Figures and aggregated CSV results are committed beside each notebook as
reproduction evidence. See [examples/README.md](examples/README.md) for runtimes
and conventions.

### Running Examples

```bash
pip install -e ".[examples]"                # tabular notebooks
pip install -e ".[examples-deep-learning]"  # plus the IMDb ablation
jupyter lab examples/
```

Launch each notebook with its own directory as the working directory; `results/`
and `figures/` are written beside it.

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
downloads its own data and uses at least five seeds.

---

## API Reference

Signatures below are the live defaults. `config/settings.yaml` carries the same
values in machine-readable form and is held to the Python signatures by
`tests/test_settings_contract.py`.

### Core functions

```python
from imv import ll, get_w, calculate_imv, information_deficit

ll(x, p, epsilon=1e-9)                       # geometric mean Bernoulli likelihood
get_w(a, guess=0.5, bounds=[(0.5, 1 - 1e-12)], tolerance=1e-9,
      chance_tolerance_nats=0.5, method="brentq")
calculate_imv(y_basic, y_enhanced, y, epsilon=1e-9, tolerance=1e-9,
              method="brentq")
information_deficit(a)                       # log(2a) nats; defined below chance
```

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
  lr=2e-5, optimizer_class=None, scheduler_fn=None, seed=None, verbose=True)`
- `calculate_imv_matrix(predictions_dict, target_column='True Label',
  prob_column='Positive Probability')`: directional matrix, rows = enhanced
- `average_imv_matrices(matrices_list)`: mean across seeds or runs
- `reduce_bert_layers(model, num_layers_to_keep)`: truncates **in place**

`calculate_imv_matrix`, `average_imv_matrices` and `reduce_bert_layers` are
static. Only constructing `AblationIMV` and training need PyTorch, so scoring
saved prediction frames works without the `deep-learning` extra.

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
- Subsample rows; the examples cap at 4,000 for exactly this reason

### `ImportError: No module named imv`
The sources live under `src/`, so the package must be installed — a dependency
list alone is not enough:

```bash
pip install -e ".[dev]"
```

### `BelowChanceLikelihoodWarning` and `NaN` results
A geometric mean likelihood below 0.5 has no equivalent coin weight, so `get_w`
returns `NaN`. This is a certificate of miscalibration, not of weak
discrimination: recalibrate the probabilities on held-out data, and use
`information_deficit(a)` to report how far below chance the predictions fall.

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
