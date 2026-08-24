# Configuration Reference

Python signatures are authoritative. `config/settings.yaml` mirrors defaults in
machine-readable form and records recommended profiles, but the package does not
load that YAML automatically. Pass non-default values explicitly to constructors
and functions.

## Core metric

| Parameter | Default | Used by |
|---|---:|---|
| `epsilon` | `1e-9` | `ll`, probability-mode IMV functions |
| `method` | `"brentq"` | `get_w` and all IMV entry points |
| `bounds` | `[(0.5, 0.999999999999)]` | `get_w` only |
| `guess` | `0.5` | Legacy `lbfgsb` backend only |
| `tolerance` | `1e-9` | Legacy `lbfgsb` backend only |
| `chance_tolerance_nats` | `0.5` | `get_w` below-chance boundary policy |

High-level IMV functions expose `epsilon`, `tolerance`, and `method`. They do not
expose custom weight bounds or below-chance tolerance; call `ll` and `get_w`
directly if an audit requires those low-level controls.

## `BinaryIMV`

| Parameter | Default | Meaning |
|---|---:|---|
| `split_method` | `"kfold"` | Shuffled K-fold parity mode |
| `n_splits` | `5` | Number of folds |
| `prop_test` | `0.2` | Holdout fraction in split modes |
| `model_type` | `"classification"` | Only accepted model type |
| `all_combinations_imv` | `None` | Optional precomputed coalition mapping |
| `random_seed` | `42` | Split random state |
| `n_jobs` | `1` | joblib workers across coalitions |
| `verbose` | `False` | Progress and summary output |

Supported split methods are `kfold`, `stratified_kfold`, `train_test_split`, and
`stratified_train_test_split`. The default preserves original-notebook behavior;
`stratified_kfold` is usually preferable for new imbalanced i.i.d. analyses.

## `MulticlassIMV`

| Parameter | Default | Meaning |
|---|---:|---|
| `n_splits` | `10` | Number of folds |
| `optional_explanatory_variables` | `None` | Use all columns except the outcome |
| `random_state` | `None` | Fold random state |
| `stratified` | `False` | Preserve original shuffled `KFold` |
| `verbose` | `False` | Print fold summaries |

Set `stratified=True` and a fixed `random_state` for a new ordinary multiclass
analysis unless a prespecified design requires otherwise.

## `AblationIMV`

The constructor defaults to `random_seed=42` and selects devices in CUDA, MPS,
CPU order. Training defaults are three epochs, learning rate `2e-5`,
`torch.optim.Adam`, no scheduler, constructor seed, and verbose output.

Prediction frames default to columns `True Label` and `Positive Probability`.
Pass `target_column` and `prob_column` when using another schema.

## Plotting

Shared heatmaps default to size `(6, 6)`, coolwarm colors, and three-decimal cell
annotations. `save_figure` always emits PNG, PDF, and SVG and defaults to 800 DPI
with `bbox_inches="tight"`.

## Cache environment variables

The library itself does not require a cache. Repository notebooks observe:

| Variable | Default |
|---|---|
| `IMV_CACHE_HOME` | `~/.cache/imv` |
| `IMV_DATA_CACHE` | `$IMV_CACHE_HOME/datasets` |
| `IMV_ARTIFACT_CACHE` | `$IMV_CACHE_HOME/notebook_artifacts` |

Set them before starting Jupyter so all notebook kernels inherit the values.

## Reproducibility profiles

`config/settings.yaml` records `paper_parity` and `recommended_production`
profiles. They are documentation, not runtime presets. The parity profile keeps
historical split choices where possible; the production profile recommends
stratification, five repeated seeds for ablation, directional matrices, and
aligned test rows. Copy only the settings justified by the current analysis.

