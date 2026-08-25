# Dry Bean Multiclass IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_dry_bean.ipynb)
provides a seven-class, continuous-feature multiclass example.

## Download and provenance

The notebook dynamically calls `ucimlrepo.fetch_ucirepo(id=602)` for the
[UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
The repository contains no raw or processed copy. Generated result tables and
figures are sent to `IMV_ARTIFACT_CACHE/multi_imv_dry_bean`.

## Preprocess

The seven bean variety names are sorted to define a stable probability-column
order. All 16 supplied geometric and shape measurements are converted to
numeric, standardized, and complete cases retained. Standardization is necessary
for the logistic model to converge on the very different input scales.

Because the scaler is fit before cross-validation, it sees aggregate held-out
statistics. This demonstration shortcut is optimistic; a new analysis should
fit the scaler separately inside each training fold.

## Models and seeds

For seeds 42 through 51, the notebook evaluates logistic regression, XGBoost,
and LightGBM with five stratified folds. `MulticlassIMV` trains a fresh
constant-only baseline and all-feature enhanced model in every fold. Tree models
are deliberately small and single-threaded so the full repeated analysis remains
executable on a typical machine.

## IMV outputs

The output preserves one-vs-rest IMV for all seven classes by estimator and seed,
plus one fold-averaged pairwise matrix per estimator/seed run. The final matrix
for each estimator is the seed mean. Missing fold contrasts would be represented
as `NaN`, although stratification and this dataset's class counts normally avoid
them.

Three symmetric pairwise heatmaps are exported as 800-DPI PNG, PDF, and SVG.
Their cells quantify improved discrimination for one bean pair at a time; they
are not directional comparisons between class labels.
