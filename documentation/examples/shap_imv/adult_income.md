# Adult Income SHAP-IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_adult_income.ipynb)
computes exact global SHAP-IMV on the UCI Adult dataset.

## Download and provenance

The notebook calls `ucimlrepo.fetch_ucirepo(id=2)` at execution time. The
[UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) is never read
from a repository path or committed to Git. Generated result tables and figure
exports go to `IMV_ARTIFACT_CACHE/shap_imv_adult_income`, defaulting beneath
`~/.cache/imv/notebook_artifacts`.

## Preprocess

The supplied income label is stripped and its possible trailing period removed;
the binary target is `income == ">50K"`. The six-feature coalition universe is
`age`, `education-num`, `hours-per-week`, `capital-gain`, `sex_female`, and
`married`. The last two are derived from `sex` and `marital-status`.

Columns are converted to numeric and incomplete rows are removed. A seeded,
class-aware subsample caps the demonstration at 4,000 rows. Features are
standardized before cross-validation so logistic regression converges. That
aggregate scaling sees held-out rows and is disclosed optimistic leakage; new
research should place `StandardScaler` inside each estimator pipeline.

## Models and seeds

For each seed in `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`, the notebook uses five-fold stratified
cross-validation and fits logistic regression, XGBoost, and LightGBM. Each
factory receives the seed; boosted estimators use 60 depth-three trees and one
thread to avoid nested parallelism.

Six features imply 64 coalitions. Every estimator/seed run therefore performs
`64 * 5 * 2 = 640` model fits through `BinaryIMV.run_evaluation`.

## IMV outputs

For every estimator and seed, the notebook retains exact SHAP-IMV for all six
features and the full-coalition IMV. It aggregates feature means and seed
standard deviations, then visualizes the attribution and full-model comparison.
Negative SHAP-IMV means a feature reduced held-out coalition performance on
average; it does not identify the negative income class.

The figure is saved as 800-DPI PNG, PDF, and SVG through the package helper.
