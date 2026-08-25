# Examples

Ten self-contained notebooks. Every one runs from a **cold clone**: no dataset is
stored in this repository, and each notebook downloads what it needs into a cache
outside the working tree.

Each notebook is a complete pipeline — download, preprocess, fit, score IMV — with
no shared helper modules, so it can be read and run in isolation.
The download call is executable notebook code, not a manual setup step, and no
notebook reads a dataset from a repository-relative path. Provider caches live
outside the working tree and may be reused on later runs.

Every notebook imports the installable `imv` package directly and fails fast unless
`imv.__file__` resolves to this checkout's `src/imv`. The ablation notebooks also
delegate seeding, device selection, training, prediction formatting, matrix
calculation, and matrix averaging to `AblationIMV`; they do not carry notebook-local
copies of those package implementations.

Notebook output never embeds a machine-specific absolute path. Package provenance
is relative to the repository, dataset messages are relative to the data cache,
figure dictionaries are relative to the artifact directory, and warning locations
are relativized before they reach notebook output.

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
| `ablation_imv_imdb.ipynb` | HF `stanfordnlp/imdb` | **~4 h accelerated; several hours CPU-only** (50 DistilBERT runs) |
| `ablation_imv_mnist.ipynb` | OpenML `mnist_784` v1 | ~10 min (50 one-epoch CNN runs, 4 CPU threads) |
| `ablation_imv_har.ipynb` | UCI HAR id 240 | ~14 min (50 15-epoch recurrent/MLP runs, 4 CPU threads) |

## Running them

```bash
pip install -r requirements.txt                # run from the repository root
jupyter lab
```

Rendered tables and figures are embedded in each executed notebook. Optional CSV
and image exports, downloaded datasets, and restart predictions are written only
under `~/.cache/imv`; set `IMV_CACHE_HOME`, `IMV_DATA_CACHE`, or
`IMV_ARTIFACT_CACHE` to override those external locations.

Every figure is exported through `imv.utils.save_figure` as an 800-DPI PNG plus
vector PDF and SVG siblings. These files remain in the external artifact cache;
the rendered notebook output is the only figure representation committed here.

The ablation notebooks select CUDA, then Apple Silicon MPS, then CPU
automatically, so they run unchanged across platforms. The IMDb configuration
(5,000 train rows, 256 tokens, 2 epochs) is a documented compute budget rather
than the published setting — see the documentation for why bit-exact replication
is not achievable on any other machine. MNIST uses its canonical 60,000/10,000
split and one epoch per fit. HAR uses UCI's subject-disjoint 21-participant/
9-participant split and 15 epochs per fit.

## These are our own runs

They are **not** replications of the published figures. Only Adult Income, Nursery
and IMDb have generating code in the original research notebooks; the remaining
datasets appear nowhere in them. Provenance is recorded per dataset in the
maintainers' documentation.

## What each notebook does to its data

Every notebook prints its raw shape and class distribution before cleaning, and
its cleaned shape after, so the preprocessing is auditable from the stored
outputs without re-running anything.

| Notebook | Target construction | Notable feature handling |
|---|---|---|
| Adult Income | `income == ">50K"`, after stripping the trailing `.` UCI's test file adds | 4 supplied numerics + `sex_female` and `married` derived |
| Titanic | `survived` | `Age`/`Fare` median-imputed; `AgeClass` interaction; `Title` from the name field |
| Breast Cancer | diagnosis `== "M"` | 6 of 30 columns (the least collinear "mean" measurements) |
| Wine Quality | `quality >= 6` — a stated choice, not a property of the data | 6 of 11 chemical properties; red and white pooled |
| Nursery | 3 of 5 classes; `recommend` (2 rows) and `very_recom` (328) dropped | all 8 categoricals → integer codes |
| Car Evaluation | 4 classes in explicit order `unacc < acc < good < vgood` | all 6 categoricals → integer codes |
| Dry Bean | 7 classes, alphabetical (no natural order) | all 16 numerics standardised |
| IMDb | supplied `label` | class-balanced 5,000/5,000 subsample, truncated to 256 tokens |
| MNIST | `digit % 2` (odd vs even) | canonical 60,000/10,000 split; pixels use standard MNIST normalization |
| HAR | `WALKING_UPSTAIRS` vs `WALKING_DOWNSTAIRS` | 128x9 inertial windows; official subject-disjoint split; train-only channel scaling |

Two example-grade shortcuts apply and are not recommendations:

- **Imputation and scaling are fitted before splitting**, so each held-out fold is
  scored by a model whose preprocessing saw it. The leak is aggregate statistics
  only, but it runs in the optimistic direction. A strict analysis passes a
  scikit-learn `Pipeline` as `model_creator` so preprocessing refits per fold.
- **Nursery and Car Evaluation encode ordinal features alphabetically**
  (`pd.Categorical(...).codes`), so `safety: low < med < high` becomes
  `1, 2, 0`. This costs logistic regression real signal; the tree models are
  largely unaffected.

The public pages under `documentation/examples/` carry the full comparative
treatment, including exact row counts and the reasoning behind each choice.

## Conventions

- **Ten complete runs per estimator or architecture** (seeds 42–51). Every
  tabular estimator and every ablation variant is fitted under each seed.
  Reported values are means; the spread describes stability and is **not** a
  confidence interval.
- **Three estimator families** — logistic regression, XGBoost, LightGBM.
- Features are standardised wherever an unscaled input would otherwise leave
  logistic regression unconverged.
- Exact SHAP-IMV costs `2**n_features * n_splits * 2` fits per seed per estimator,
  so feature counts and row counts are capped deliberately.

## Interpreting the output

- A negative SHAP-IMV value means the feature **reduced** held-out information
  (overfitting or calibration damage). It does **not** indicate the negative class.
  SHAP-IMV is a global metric attribution and does not use the `shap` library.
- IMV comparisons are **directional**: the baseline and enhanced predictions
  have distinct roles, so reversing them can change the result. This applies to
  the binary, multiclass, and ablation extensions.
- A likelihood below 0.5 has no equivalent coin: `get_w` returns `NaN` and
  `information_deficit` reports the shortfall in nats. Treat it as evidence of
  miscalibration, not of weak discrimination.
