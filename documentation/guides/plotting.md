# Plotting and Export

The shared plotting functions accept either an existing Matplotlib axis or
create a new figure. This supports both interactive use and multi-panel
publication figures.

## Heatmaps

```python
from imv.utils import plot_ablation_matrix, plot_imv_heatmap

figure, axis = plot_imv_heatmap(
    matrix,
    title="Pairwise class IMV",
    fmt=".3f",
)

figure_ablation, axis_ablation = plot_ablation_matrix(ablation_matrix)
```

Matrices must be non-empty square two-dimensional arrays. A pandas DataFrame
supplies labels from its columns automatically; otherwise provide `labels`.
`plot_ablation_matrix` is the same heatmap with an ablation-specific title.

## One-vs-rest distributions

```python
from imv.utils import plot_ova_boxplot

figure, axis = plot_ova_boxplot(
    fold_scores,
    labels=["class A", "class B", "class C"],
    title="One-vs-rest stability",
)
```

`fold_scores` is a non-empty two-dimensional folds-by-classes matrix. Box width
or fold spread describes sensitivity to the fold partition, not a confidence
interval.

## Existing axes

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 2, figsize=(12, 5))
returned = plot_imv_heatmap(matrix_a, ax=axes[0], title="Estimator A")
plot_imv_heatmap(matrix_b, ax=axes[1], title="Estimator B")

assert returned is axes[0]
```

When `ax=None`, plotting functions return `(figure, axis)`. When `ax` is
provided, they return that axis.

## Publication export

```python
from imv.utils import FIGURE_DPI, FIGURE_FORMATS, save_figure

paths = save_figure(figure, "artifacts/result.png")

assert FIGURE_DPI == 800
assert FIGURE_FORMATS == ("png", "pdf", "svg")
print(paths)
```

The destination may have no suffix or any supported suffix. `save_figure`
removes that suffix and writes three siblings:

```text
artifacts/result.png
artifacts/result.pdf
artifacts/result.svg
```

PNG is written at 800 DPI. The DPI is also passed to PDF and SVG so rasterized
artists inside vector figures use the same resolution. Parent directories are
created automatically, and the returned dictionary maps format name to `Path`.

The caller controls the output directory. Keep generated figures outside the
package source tree unless they are intentional documentation assets.
