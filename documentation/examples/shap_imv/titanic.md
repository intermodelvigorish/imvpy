# Titanic SHAP-IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_titanic.ipynb)
computes exact global SHAP-IMV for passenger survival.

## Download and provenance

`sklearn.datasets.fetch_openml("titanic", version=1, as_frame=True)` downloads
the [OpenML Titanic dataset](https://www.openml.org/d/40945) on demand. OpenML's
cache is directed to `IMV_DATA_CACHE/openml`, defaulting beneath
`~/.cache/imv/datasets`. No dataset is stored in the repository.

## Preprocess

The binary target is the supplied `survived` field. The six features are `Sex`,
`Title`, `Class`, `AgeClass`, `Fare`, and `Embarked`. Age and fare are
median-imputed; `AgeClass` multiplies age by passenger class; embarkation is
integer-mapped. Titles are parsed from names, rare titles are pooled, and the
title ordering is explicitly the notebook authors' choice because the published
encoding was not recorded.

Rows are numeric and complete after these transformations. The 4,000-row cap is
inactive for this dataset. Features are standardized before splitting, which
leaks aggregate held-out statistics; a new analysis should fit imputation,
encoding, and scaling within each fold.

## Models and seeds

Seeds `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]` each define a five-fold stratified partition and
the random state for logistic regression, XGBoost, and LightGBM. Six features
produce 64 coalitions and 640 null/enhanced fits per estimator per seed.

## IMV outputs

All coalition values are calculated by `BinaryIMV`, then each feature's exact
SHAP-IMV and the full-model IMV are retained by estimator and seed. The notebook
plots means with seed standard deviations and exports PNG at 800 DPI plus PDF
and SVG. Title and sex are intentionally both present, so their redundancy is
visible in coalition attribution rather than hidden by pre-selection.
