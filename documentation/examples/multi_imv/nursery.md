# Nursery Multiclass IMV

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/multi_imv/multi_imv_nursery.ipynb)
demonstrates one-vs-rest and pairwise multiclass IMV.

## Download and provenance

The notebook downloads the [UCI Nursery dataset](https://archive.ics.uci.edu/dataset/76/nursery)
with `ucimlrepo.fetch_ucirepo(id=76)` at runtime. It never reads a repository
data file. CSV and figure outputs are written to
`IMV_ARTIFACT_CACHE/multi_imv_nursery` outside the checkout.

## Preprocess

The two very small classes `recommend` (2 rows) and `very_recom` (328 rows) are
removed because repeated multiclass folds cannot estimate them reliably. The
remaining explicit class order is `not_recom`, `priority`, `spec_prior`.
All eight supplied categorical features are converted with pandas category
codes, then complete rows are retained.

Alphabetical category codes do not preserve the documented ordinal meaning of
every feature. This can cost logistic regression signal, while the tree models
are less sensitive. A research analysis should encode categories from their data
dictionary inside a fold-fitted pipeline.

## Models and seeds

For every seed in `[42, 43, 44, 45, 46]`, `MulticlassIMV` runs five stratified
folds for logistic regression, XGBoost, and LightGBM. Each seed controls fold
shuffling and estimator randomness. The enhanced model uses all eight features;
the package fits its constant-only baseline in the same training fold.

## IMV outputs

The notebook retains one-vs-rest IMV for every class, estimator, and seed. It
also averages the package's fold-level pairwise matrix within each run and then
across seeds. The pairwise matrix is symmetric by construction; it should not be
read like a directional ablation matrix.

The final multi-panel heatmap compares the three estimator families and is
exported as 800-DPI PNG, PDF, and SVG. Seed spread indicates stability rather
than inferential uncertainty.

