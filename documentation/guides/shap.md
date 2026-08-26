# Exact SHAP-IMV

`BinaryIMV` treats the held-out IMV of a feature coalition as a cooperative game
value and computes the exact Shapley attribution of that value.

!!! note
    SHAP-IMV does not import or call the separate `shap` package. It is a global
    attribution of predictive performance across feature coalitions, not a local
    explanation of one prediction.

## Estimator contract

`model_creator` must return a fresh, unfitted binary classifier implementing
`fit(X, y)` and `predict_proba(X)` with positive-class probability in column 1.
Put learned preprocessing in a scikit-learn `Pipeline` so it is refit inside
each fold.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imv import BinaryIMV

X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)
features = ["x1", "x2", "x3", "x4"]
data = pd.DataFrame(X, columns=features).assign(target=y)

def model_creator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )

evaluator = BinaryIMV(
    data=data,
    outcome_variable="target",
    optional_explanatory_variables=features,
    model_creator=model_creator,
    split_method="stratified_kfold",
    n_splits=5,
    random_seed=42,
    n_jobs=1,
    verbose=False,
)
```

The input DataFrame is copied. Internally the evaluator adds a constant column
for the null model, so an existing feature named `constant` should not be used.
Missing values are rejected; impute them inside `model_creator`.

## Fit every coalition

```python
coalitions = evaluator.run_evaluation()
```

The return value and `evaluator.all_combinations_imv` are the same mapping:

```text
{
    feature_tuple: (mean_imv_across_folds, [fold_imv, ...]),
    ...
}
```

The empty tuple is the null coalition. Feature tuples preserve the order supplied
to the constructor.

Exact evaluation costs:

```text
2**n_features * n_splits * 2 model fits
```

The factor of two is the separately fitted null and enhanced model. Cost grows
exponentially in feature count; `n_jobs` parallelizes coalitions but does not
change the number of fits. Avoid nested parallelism by keeping estimator-level
thread counts at one when `BinaryIMV.n_jobs` is greater than one.

## Calculate Shapley values

```python
values = {
    feature: evaluator.calculate_imvshapley_value(feature)
    for feature in features
}
```

For feature `j`, the implementation sums over every subset `S` that excludes it:

```text
phi_j = sum(|S|! * (n-|S|-1)! / n! * (v(S union {j}) - v(S)))
```

Returned feature values are rounded to three decimal places. Before rounding,
exact Shapley additivity relates their sum to the full-coalition value minus the
empty-coalition value. The rounded sum can differ slightly.

If `all_combinations_imv` is supplied manually and lacks required coalitions,
`calculate_imvshapley_value` emits `IncompleteCoalitionWarning`. Missing values
are substituted with zero only for backward compatibility; the result is not a
valid Shapley value and should not be reported.

## Plot and export

```python
from pathlib import Path

from imv.utils import save_figure

figure, axis = evaluator.evaluate_imvshapley(figsize=(10, 4))
paths = save_figure(figure, Path("outputs") / "shap_imv")
```

`evaluate_imvshapley` returns `(figure, axis)` when it creates the axis, or the
provided axis when composing a larger figure. The single-variable violin method
uses the same return convention.

## Split modes

| `split_method` | Behavior |
|---|---|
| `stratified_kfold` | Recommended shuffled stratified folds for ordinary i.i.d. binary data |
| `kfold` | Shuffled unstratified folds, retained for legacy parity |
| `stratified_train_test_split` | One stratified holdout using `prop_test` |
| `train_test_split` | One unstratified holdout, retained for parity |

For grouped or temporal data, construct held-out coalition predictions with a
custom workflow rather than using an invalid shuffled split.
