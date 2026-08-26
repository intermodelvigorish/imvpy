# Statistical Practice

The package can calculate a number correctly while the analysis design remains
invalid. These practices are part of a defensible IMV analysis.

## Use honest predictions

Generate every probability on observations excluded from that model's fit.
Cross-fitted, out-of-fold, or genuinely external test predictions are suitable.
In-sample probabilities favor flexible enhanced models and invalidate the model
comparison.

Fit preprocessing inside each training fold. Imputation, scaling, feature
selection, target encoding, calibration, and hyperparameter tuning can all leak
information if performed before splitting. A scikit-learn `Pipeline` is the
safest `model_creator` for built-in evaluators.

## Match the split to the data

- Use stratification for imbalanced i.i.d. classification unless reproducing a
  prespecified non-stratified design.
- Split by participant, site, household, or other dependence group when rows are
  not independent.
- Respect temporal order for forecasting and deployment-over-time questions.
- Nest tuning and calibration inside the outer evaluation loop.

The built-in evaluators provide shuffled K-fold and stratified K-fold variants,
not every possible design. For grouped or temporal designs, create predictions
externally and use the core or ablation scoring functions.

## Keep comparisons aligned

Both predictors must score the same outcomes in the same order. The ablation
matrix API enforces identical target columns and row counts, but it cannot infer
whether two equal labels belong to the same underlying observation. Preserve a
stable identifier in your own workflow and check it before dropping that column
for scoring.

## Treat probabilities as measurements

IMV uses the full probability scale. Two models with equal ranking or accuracy
can have different IMV because their confidence differs. Assess calibration on
training-only or nested validation data, then apply the selected calibrator to
the held-out rows. Never recalibrate directly on the final test outcomes.

## Repeat stochastic fits

Neural networks, boosted trees, subsampling, and shuffled folds can vary by seed.
Run at least ten complete seeds when seed variability is material. Repeat the
entire pipeline, not only the final metric calculation, and retain seed-level
results before averaging.

The standard deviation across ten seeds describes run-to-run stability. It is
not a confidence interval for a population quantity. Folds also overlap in their
training data, so their spread is not an independent-sample standard error.

## Report enough to reproduce

Report all of the following with an IMV result:

- Package version and inverse backend.
- Outcome definition and positive class.
- Dataset source, retrieval version/date, exclusions, and preprocessing.
- Baseline and enhanced model definitions.
- Train/test or cross-validation design, including grouping or stratification.
- Hyperparameter selection and probability calibration procedure.
- Seeds and whether the reported value is per-run, mean, or another summary.
- Number of observations evaluated and proof of row alignment for model ablation.
- Below-chance warnings or undefined values rather than silently removing them.
- For SHAP-IMV, the complete feature universe and whether the full power set was
  evaluated.
