# Model Ablation

Ablation IMV compares binary probability predictions from model variants on the
same observations. The variants can be neural or non-neural and can be trained
by any framework.

## Score existing predictions without PyTorch

Matrix scoring is static and does not require constructing `AblationIMV`:

```python
import pandas as pd

from imvpy import AblationIMV

predictions = {
    "Full": pd.DataFrame({
        "True Label": [1, 0, 1, 0],
        "Positive Probability": [0.90, 0.10, 0.80, 0.20],
    }),
    "NoAttention": pd.DataFrame({
        "True Label": [1, 0, 1, 0],
        "Positive Probability": [0.72, 0.25, 0.66, 0.30],
    }),
    "Linear": pd.DataFrame({
        "True Label": [1, 0, 1, 0],
        "Positive Probability": [0.58, 0.43, 0.55, 0.46],
    }),
}

matrix = AblationIMV.calculate_imv_matrix(predictions)
```

Rows are enhanced models and columns are baseline models. Cell `(i, j)` is:

```text
IMV(model j -> model i)
```

Read down the `Linear` column to compare all enhanced variants against that one
fixed baseline. Because each column has a different denominator, the matrix is
directional and generally not antisymmetric.

Every frame must contain the requested target and probability columns. Labels
must be identical and aligned across frames. Preserve and verify your own row ID
before constructing these two-column frames; equal label sequences alone cannot
prove observation identity.

## Average complete seed-level matrices

Train and score every variant under each seed, then average whole matrices:

```python
seed_matrices = []

for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
    predictions_for_seed = train_all_variants(seed)
    seed_matrices.append(
        AblationIMV.calculate_imv_matrix(predictions_for_seed)
    )

mean_matrix = AblationIMV.average_imv_matrices(seed_matrices)
```

All matrices must be pandas DataFrames with identical index and column order.
The function performs an elementwise arithmetic mean. Compute and retain
seed-level dispersion separately; it is not a confidence interval.

Do not combine a full model from one seed with an ablation from another seed in
the same seed-level contrast. Shared seeds control one source of run-to-run
variation, while repeating at least ten complete seeds exposes sensitivity to
initialization and minibatch order.

## PyTorch training helper

Install `imvpy[deep-learning]`, then construct the evaluator. Device priority is
CUDA, Apple MPS, then CPU.

```python
from imvpy import AblationIMV

ablator = AblationIMV(random_seed=42)
result = ablator.train_and_evaluate(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    num_epochs=3,
    lr=2e-5,
    seed=42,
    verbose=True,
)

prediction_frame = result["test_predictions"]
```

The trainer expects each batch to be a mapping of tensors, including a `labels`
key. It calls `model(**batch)` and expects an output with scalar `.loss` and
two-column `.logits`. It is therefore directly compatible with binary Hugging
Face sequence classifiers and with custom PyTorch modules that expose the same
interface.

The result dictionary contains the trained model, aligned prediction DataFrame,
accuracy, binary precision, and binary recall. The prediction frame columns are
`Negative Probability`, `Positive Probability`, `True Label`, and
`Predicted Label`.

If `optimizer_class` is omitted, the trainer uses `torch.optim.Adam`. A custom
`scheduler_fn` is called with keyword arguments `optimizer` and
`num_training_steps` and must return an object with `.step()`.

## DistilBERT layer reduction

```python
from transformers import DistilBertForSequenceClassification

fresh = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
three_layers = AblationIMV.reduce_bert_layers(fresh, 3)
```

`reduce_bert_layers` mutates the supplied model in place. Construct a fresh model
for every variant; otherwise later surgery contaminates earlier variants. The
helper specifically accesses `model.distilbert.transformer.layer`; other
transformer families need architecture-specific surgery.

## Plot and export

```python
from imvpy.utils import plot_ablation_matrix, save_figure

figure, axis = plot_ablation_matrix(mean_matrix)
paths = save_figure(figure, "outputs/ablation_matrix")
```

If any variant has a geometric mean likelihood substantially below 0.5, its
weight and affected matrix cells are undefined. Report its
`information_deficit` and investigate calibration rather than forcing those
cells to zero.
