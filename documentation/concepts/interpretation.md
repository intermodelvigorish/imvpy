# Interpretation

## What an IMV value says

IMV is the relative change in equivalent-coin weight from the declared baseline
to the declared enhanced predictor. For example, `0.10` means the enhanced
weight is 10% larger than the baseline weight. It does not mean ten percentage
points of accuracy, ten percent of outcome variance, or ten percent of mutual
information.

Always retain the arrow in prose: “IMV from the intercept-only model to model X”
is less ambiguous than “model X has IMV 0.10.”

## Baseline dependence

The denominator is the baseline weight. Two studies using different baselines
can obtain different IMV magnitudes for the same enhanced predictions. Compare
values only when the outcome definition, evaluation rows, and baseline are
compatible.

A constant prevalence model is a useful null baseline when it is estimated only
on training data. A domain model, an earlier production model, or an ablated
architecture can also be valid if it matches the scientific contrast.

## Negative values

A negative IMV means the enhanced predictor assigned less geometric mean
likelihood to held-out outcomes than the baseline. Possible causes include
overfitting, harmful features or components, poor optimization, probability
miscalibration, distribution shift, or sampling variability.

For SHAP-IMV, a negative feature attribution means the feature reduced the
global coalition value on average. It does not mean the feature predicts the
negative outcome class.

## Below-chance results

A substantially below-0.5 geometric mean likelihood has no equivalent coin and
therefore no IMV weight. Treat the resulting warning and `NaN` as diagnostic
information, not a value to replace with zero. Inspect calibration and data
alignment, and report `information_deficit` when the severity itself matters.

## Matrix semantics

| Output | Row role | Column role | Symmetry |
|---|---|---|---|
| `AblationIMV.calculate_imv_matrix` | Enhanced model | Baseline model | Directional; generally neither symmetric nor antisymmetric |
| `MulticlassIMV.multinominal_imv_matrix` | First class in a pair | Second class in a pair | Symmetric by construction |

Read an ablation matrix down one baseline column when comparing variants against
that fixed baseline. Do not rank cells across columns as if their denominators
were the same.

## What IMV is not

- IMV is not mutual information, even though it uses an entropy-shaped inverse.
- IMV is not itself a calibration score; it is sensitive to calibration because
  it consumes probabilities.
- IMV is not defined by hard class predictions or accuracy.
- This implementation is not a regression metric.
- Fold or seed dispersion is not automatically a standard error or confidence
  interval.

