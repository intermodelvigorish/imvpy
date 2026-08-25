# Wine Quality SHAP-IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_wine_quality.ipynb)
turns the ordinal UCI quality score into a stated binary example outcome.

## Download and provenance

`ucimlrepo.fetch_ucirepo(id=186)` downloads the
[UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
during execution. No raw or processed dataset is committed. Artifact exports are
kept under the external `IMV_ARTIFACT_CACHE`.

## Preprocess

The notebook defines the positive class as `quality >= 6`; this threshold is an
analysis choice, not a property of UCI. Its six features are `alcohol`,
`volatile_acidity`, `sulphates`, `total_sulfur_dioxide`, `density`, and
`residual_sugar`, with a documented first-six-numeric fallback for naming
differences.

Numeric conversion and complete-case filtering precede a seeded, class-aware
4,000-row cap. Features are standardized before folds so logistic regression
converges. This convenience exposes held-out aggregate statistics, so a strict
analysis should move scaling into `model_creator`.

## Models and seeds

The complete six-feature power set is evaluated for logistic regression,
XGBoost, and LightGBM. Each estimator is rerun for seeds `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`
with five stratified folds, producing 640 model fits per estimator and seed.

## IMV outputs

`BinaryIMV` supplies every coalition value and feature SHAP-IMV. The notebook
keeps the estimator and seed dimensions, summarizes mean and standard deviation,
and compares feature attribution with full-model IMV. The standard deviation is
a seed-sensitivity summary, not a confidence interval. Figures are exported as
800-DPI PNG, PDF, and SVG.
