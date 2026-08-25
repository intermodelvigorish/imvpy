# Executed Examples

The repository contains ten fully executed notebooks. Every notebook downloads a
public dataset dynamically, imports this checkout's installed `imv` package, runs
every stochastic model or architecture under seeds 42 through 51, and embeds its
tables and figures.

| Family | Dataset | Model diversity | Notebook |
|---|---|---|---|
| Exact SHAP-IMV | Adult Income | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_adult_income.ipynb) |
| Exact SHAP-IMV | Titanic | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_titanic.ipynb) |
| Exact SHAP-IMV | Breast Cancer | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_breast_cancer.ipynb) |
| Exact SHAP-IMV | Wine Quality | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/shap_imv/shap_imv_wine_quality.ipynb) |
| Multiclass | Nursery | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_nursery.ipynb) |
| Multiclass | Car Evaluation | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_car_evaluation.ipynb) |
| Multiclass | Dry Bean | Logistic regression, XGBoost, LightGBM | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_dry_bean.ipynb) |
| Ablation | IMDb | DistilBERT transformer components | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_imdb.ipynb) |
| Ablation | MNIST odd/even | Convolutional architecture components | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_mnist.ipynb) |
| Ablation | UCI HAR upstairs/downstairs | Recurrent sequence architecture components | [Open notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_har.ipynb) |

## Shared execution contract

- No dataset or downloaded model is committed. Provider data caches live outside
  the working tree.
- Seeds `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]` wrap every estimator or architecture that has seed
  variability.
- Tabular examples compare three estimator families rather than presenting one
  model as universal.
- Ablation examples execute five variants under ten seeds, for 50 fits each.
- All IMV scoring delegates to public functions or classes from this package.
- Every figure is embedded and exported through `imv.utils.save_figure` as an
  800-DPI PNG, PDF, and SVG in the external artifact cache.
- Seed standard deviations are presented as stability summaries, not confidence
  intervals.

Install and run them as described in [Installation](../getting-started/installation.md).

## Methodological status

These notebooks are demonstrations on commonly used open datasets. They are not
benchmark leaderboards. Only Adult Income, Nursery, and IMDb have generating
code in the original research notebooks; none of these repository runs should be
described as a bit-exact replication of a published figure.
