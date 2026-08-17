# Examples

Eight self-contained notebooks. Every one runs from a **cold clone**: no dataset is
stored in this repository, and each notebook downloads what it needs into a cache
outside the working tree.

Each notebook is a complete pipeline — download, preprocess, fit, score IMV — with
no shared helper modules, so it can be read and run in isolation.

## Binary — exact SHAP-IMV (`shap_imv/`)

| Notebook | Source | Runtime |
|---|---|---|
| `shap_imv_adult_income.ipynb` | UCI id 2 | ~80 s |
| `shap_imv_titanic.ipynb` | OpenML `titanic` v1 | ~60 s |
| `shap_imv_breast_cancer.ipynb` | UCI id 17 | ~60 s |
| `shap_imv_wine_quality.ipynb` | UCI id 186 | ~80 s |

## Multiclass (`multi_imv/`)

| Notebook | Source | Runtime |
|---|---|---|
| `multi_imv_nursery.ipynb` | UCI id 76 | ~20 s |
| `multi_imv_car_evaluation.ipynb` | UCI id 19 | ~10 s |
| `multi_imv_dry_bean.ipynb` | UCI id 602 | ~50 s |

## Ablation (`ablation_imv/`)

| Notebook | Source | Runtime |
|---|---|---|
| `ablation_imv_imdb.ipynb` | HF `stanfordnlp/imdb` | **~2 h** (25 DistilBERT runs) |

## Running them

```bash
pip install -e ".[dev]"                       # package + jupyter + test extras
pip install ucimlrepo                         # UCI downloads
pip install transformers datasets             # ablation only
jupyter lab
```

Launch each notebook with its own directory as the working directory; `results/`
and `figures/` are written beside it.

The ablation notebook selects CUDA, then Apple Silicon MPS, then CPU
automatically, so it runs unchanged across platforms. Its configuration (5,000
train rows, 256 tokens, 2 epochs) is a documented compute budget rather than the
published setting — see the documentation for why bit-exact replication is not
achievable on any other machine.

## These are our own runs

They are **not** replications of the published figures. Only Adult Income, Nursery
and IMDb have generating code in the original research notebooks; the remaining
datasets appear nowhere in them. Provenance is recorded per dataset in the
maintainers' documentation.

## Conventions

- **Five seeds** (42–46) everywhere. Reported values are means; the spread
  describes stability and is **not** a confidence interval.
- **Three estimator families** — logistic regression, XGBoost, LightGBM.
- Features are standardised wherever an unscaled input would otherwise leave
  logistic regression unconverged.
- Exact SHAP-IMV costs `2**n_features * n_splits * 2` fits per seed per estimator,
  so feature counts and row counts are capped deliberately.

## Interpreting the output

- A negative SHAP-IMV value means the feature **reduced** held-out information
  (overfitting or calibration damage). It does **not** indicate the negative class.
  SHAP-IMV is a global metric attribution and does not use the `shap` library.
- The multiclass pairwise matrix is **symmetric by construction**; the ablation
  matrix **is** directional, with a different baseline in each column.
- A likelihood below 0.5 has no equivalent coin: `get_w` returns `NaN` and
  `information_deficit` reports the shortfall in nats. Treat it as evidence of
  miscalibration, not of weak discrimination.
