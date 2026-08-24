# Breast Cancer SHAP-IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_breast_cancer.ipynb)
uses the Wisconsin Diagnostic Breast Cancer data.

## Download and provenance

The notebook dynamically calls `ucimlrepo.fetch_ucirepo(id=17)` for the
[UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Nothing under the repository supplies or stores the data. Optional results and
figures are written to the external artifact cache.

## Preprocess

The binary target is `diagnosis == "M"`. Six mean-geometry measurements are
used: `radius1`, `texture1`, `smoothness1`, `compactness1`, `symmetry1`, and
`fractal_dimension1`. The notebook has a documented first-six-column fallback
for UCI mirrors with different column names.

Inputs are converted to numeric, incomplete rows are removed, and features are
standardized. The dataset is below the 4,000-row demonstration cap. As in the
other tabular examples, scaling occurs before cross-validation for readability;
fit it inside a pipeline for leakage-free research.

## Models and seeds

Logistic regression, XGBoost, and LightGBM are each run under seeds 42 through
46 with five stratified folds. Every seed/estimator combination evaluates the
complete 64-coalition power set through the installed `BinaryIMV` class.

## IMV outputs

The notebook records seed-level feature SHAP-IMV and full-model IMV, then reports
means and seed standard deviations. This dataset is small and highly separable,
so probability calibration and the high-likelihood behavior of `get_w` matter;
accuracy alone would conceal those differences. The final figure is exported in
PNG at 800 DPI, PDF, and SVG.

