# Choose a Workflow

Use the lowest-level workflow that answers the scientific question.

## Compare two binary predictors

Use [vanilla IMV](../guides/vanilla.md) when you already have aligned binary
outcomes and probabilities. The predictors may come from any model family. A
constant prevalence estimate is a common null baseline, but any probabilistic
predictor can be the baseline if it is stated explicitly.

## Attribute a binary model to features

Use [exact SHAP-IMV](../guides/shap.md) when the value of each feature should be
averaged over every feature coalition. `BinaryIMV` fits all `2**n_features`
coalitions and is therefore suitable only for a modest feature count. This is a
global model-performance attribution, not the local explanation implemented by
the separate `shap` library.

## Evaluate a multiclass model

Use [multiclass IMV](../guides/multiclass.md) for class-vs-rest values and a
pairwise class-separation matrix. The pairwise matrix is symmetric by
construction. If only one scientifically chosen contrast matters, converting it
to a binary outcome and using vanilla IMV may be simpler and more transparent.

## Compare model components or architectures

Use [ablation IMV](../guides/ablation.md) when the observations are fixed and
multiple variants produce aligned binary probabilities. The package includes a
PyTorch/Hugging Face-style trainer, but the matrix calculator accepts prediction
DataFrames from any training framework. This is the right path for CNN, RNN,
transformer, or non-neural ablations.

## Custom validation designs

For grouped, nested, blocked, or temporal validation, generate held-out
predictions with the appropriate external splitter and call `vanilla_imv` or
`AblationIMV.calculate_imv_matrix`. Do not force an invalid scientific design
into the built-in shuffled fold evaluators merely for convenience.

