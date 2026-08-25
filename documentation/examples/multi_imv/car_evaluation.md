# Car Evaluation Multiclass IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_car_evaluation.ipynb)
applies multiclass IMV to four car-acceptability classes.

## Download and provenance

`ucimlrepo.fetch_ucirepo(id=19)` retrieves the
[UCI Car Evaluation dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)
on demand. No dataset is stored in Git. Results and all figure formats remain in
the external artifact cache.

## Preprocess

The class order is explicitly `unacc`, `acc`, `good`, `vgood`. All six
categorical predictors are converted to pandas category codes and complete rows
are retained. This alphabetical encoding does not necessarily preserve domain
order, for example safety levels; it is an example shortcut and can particularly
harm the linear estimator. A production analysis should use dictionary-defined
ordinal or one-hot encoding inside each fold.

## Models and seeds

Logistic regression, XGBoost, and LightGBM are evaluated over five stratified
folds for each of seeds `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`. The seed controls both the fold
partition and estimator randomness. All estimators expose aligned
`predict_proba` columns through `classes_`, which the package passes to its
low-level fold calculations.

## IMV outputs

For every estimator and seed, the notebook records class-vs-rest IMV and the
fold-averaged pairwise IMV matrix. Class imbalance makes the rare `vgood`
contrast the least stable, so its seed spread must accompany the mean. Pairwise
matrices are symmetric and compare feature-enhanced discrimination against the
constant baseline for each class pair.

The estimator heatmaps use the shared package plotting style and are saved as
PNG at 800 DPI, PDF, and SVG.
