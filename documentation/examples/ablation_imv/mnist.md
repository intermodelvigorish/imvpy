# MNIST CNN Ablation

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_mnist.ipynb)
uses convolutional component ablations to classify odd versus even handwritten
digits.

## Download and provenance

`sklearn.datasets.fetch_openml("mnist_784", version=1)` downloads
[MNIST from OpenML](https://www.openml.org/d/554) into
`IMV_DATA_CACHE/openml`, defaulting beneath `~/.cache/imv/datasets`. The notebook
asserts the expected 70,000-row shape and never reads a repository data path.

This architecture follows the broad pattern of the linked open-source MNIST
seed-sensitivity example, but all training, prediction formatting, and IMV
scoring use this repository's installed package.

## Preprocess

Pixels are reshaped to `(70_000, 1, 28, 28)`, scaled to `[0, 1]`, and normalized
with the standard MNIST mean 0.1307 and standard deviation 0.3081. The binary
target is `digit % 2`, with odd digits as one. The canonical first 60,000 rows
form training data and the final 10,000 form test data.

Every variant in one seed scores the same fixed 10,000 test rows. Cached
prediction validation checks row count, exact labels, finite probability bounds,
and that positive plus negative probability equals one.

## Ablations and seeds

The five variants are `FullCNN`, `NoConv2`, `NoHidden`, `NoDropout`, and
`Linear`. They separately remove the second convolution, 128-unit hidden layer,
dropout, or the complete convolutional feature extractor. This gives a graded
architecture study rather than five cosmetic hyperparameter changes.

Each variant is trained under seeds `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]` through
`AblationIMV.train_and_evaluate`: 50 complete fits, one epoch each, Adadelta with
learning rate 1.0, training batch 256, and test batch 1,000. Seeds control model
initialization, shuffled minibatches, and dropout.

## IMV outputs

For every seed, `AblationIMV.calculate_imv_matrix` receives the five aligned
prediction frames. `average_imv_matrices` produces the mean directional matrix;
seed-level standard deviations are retained separately. Diagnostics include
parameter count, likelihood, information deficit, accuracy, runtime, and cache
status.

The first figure shows the full directional matrix. The second fixes each
ablation as the baseline and `FullCNN` as the enhanced model, with seed standard
deviation bars. Both are exported as 800-DPI PNG, PDF, and SVG to the external
artifact cache.
