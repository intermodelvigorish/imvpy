# Troubleshooting

## `ModuleNotFoundError: No module named 'imv'`

Install the project; the source lives under `src/`:

```bash
python -m pip install -e .
python -c "import os, imv; print(os.path.relpath(imv.__file__))"
```

If Python imports a different checkout or release, reinstall in editable mode
with the interpreter used to run the analysis.

## PyTorch import error

Core metric and static matrix scoring do not need PyTorch. Construction or
training with `AblationIMV`, seeding, and BERT layer reduction do:

```bash
python -m pip install ".[deep-learning]"
```

## `BelowChanceLikelihoodWarning` or `NaN`

A geometric mean likelihood sufficiently below 0.5 has no equivalent-coin root.
Check that labels and probabilities are aligned, that the positive class is
correct, and that probabilities are calibrated. Use
`information_deficit(likelihood)` to quantify the shortfall. Do not replace the
undefined value with zero.

## `IncompleteCoalitionWarning`

Exact SHAP-IMV requires all `2**n_features` coalitions. Call
`run_evaluation()` over the complete feature list. A value calculated after the
warning uses compatibility substitution for absent coalitions, breaks Shapley
additivity, and should not be reported.

## Length, shape, or probability errors

Core inputs must be finite one-dimensional vectors. Outcomes contain only 0/1;
probabilities lie in `[0, 1]`; both predictors match the outcome length. A
numeric scalar broadcasts, while `[0.5]` and `np.array([0.5])` are vectors and do
not broadcast.

For ablation matrices, every DataFrame must have identical aligned target
columns. For multiclass low-level methods, probability arrays have shape
`(samples, classes)` and `classes` names columns in exactly that order.

## A multiclass fold omits a class

Set `stratified=True` when class counts permit. Low-level methods return `NaN`
for unmeasurable absent-class contrasts. Always pass `classes=model.classes_`
when supplying fold probabilities yourself, because labels observed in a test
fold may not describe every probability column.

## SHAP-IMV is too slow

Cost is `2**n_features * n_splits * 2` fits. Reduce the feature universe, folds,
or estimator cost; use one holdout only when scientifically defensible; or set
`n_jobs` explicitly. Avoid estimator-level parallelism at the same time. The
package does not implement sampled coalition approximation.

## Deep-learning memory or runtime issues

Reduce batch size, sample count, token length, epochs, or variant count for an
exploratory run. Keep the final design identical across variants. If callers
cache predictions, include the package version, model configuration, and seed in
cache keys and validate row alignment before reuse.

## Documentation build fails

Install the documentation extra and use strict mode from the repository root:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

An unresolved API object usually means a documented symbol was renamed or its
module path is wrong. A navigation warning means a Markdown page is missing from
`mkdocs.yml` or vice versa; both are treated as release failures.
