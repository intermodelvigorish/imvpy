# Plotting and Export

The shared plotting functions accept either an existing Matplotlib axis or
create a new figure. This supports both interactive use and multi-panel
publication figures.

## Visual system

IMVpy uses the same visual language as the IMV research figures:

- A navy-to-steel-blue-to-cream-to-red continuous colormap registered as `imv`
  and `imv_r`.
- Helvetica typography when installed, followed by Nimbus Sans and DejaVu Sans
  as portable fallbacks.
- White backgrounds, thin dark axes and mark outlines, dashed light-gray grids,
  bold left-aligned titles, and compact `5.4 x 4.6` inch panel geometry.
- Three-decimal bar and default heatmap annotations. The lower-level
  `heatmap_style` switches to two decimals for more than five classes.

Package plotting functions apply these choices within a scoped Matplotlib
context, so they do not overwrite unrelated plotting defaults.

```python
import matplotlib.pyplot as plt

from imvpy.utils import plotting_context, spectral_colors

with plotting_context():
    figure, axis = plt.subplots()
    axis.plot([0, 1, 2], [0.1, 0.4, 0.3], color=spectral_colors(1)[0])
```

For a notebook dedicated entirely to IMV figures, opt into process-wide styling:

```python
from imvpy.utils import configure_plotting

configure_plotting()
```

`configure_plotting` intentionally modifies Matplotlib's global `rcParams`.
Prefer `plotting_context` in reusable modules or applications. Use `rc_params`
when another context manager or plotting framework needs the settings mapping.

## Heatmaps

```python
from imvpy.utils import plot_ablation_matrix, plot_imv_heatmap

figure, axis = plot_imv_heatmap(
    matrix,
    title="Pairwise class IMV",
    fmt=".3f",
)

figure_ablation, axis_ablation = plot_ablation_matrix(ablation_matrix)
```

Matrices must be non-empty square two-dimensional arrays. A pandas DataFrame
supplies labels from its columns automatically; otherwise provide `labels`.
`plot_ablation_matrix` uses an ablation-specific title and places zero at the
neutral cream midpoint. `plot_imv_heatmap` accepts `cmap`, `center`, and
`colorbar_label` when another explicit encoding is required.

The lower-level `heatmap_style`, `style_heatmap_frame`, and
`style_heatmap_axes` helpers support custom seaborn heatmaps. For a multi-panel
layout, `add_heatmap_colorbar` creates a full-height vector colorbar without
resizing sibling panels.

## One-vs-rest distributions

```python
from imvpy.utils import plot_ova_boxplot

figure, axis = plot_ova_boxplot(
    fold_scores,
    labels=["class A", "class B", "class C"],
    title="One-vs-rest stability",
)
```

`fold_scores` is a non-empty two-dimensional folds-by-classes matrix. Box width
or fold spread describes sensitivity to the fold partition, not a confidence
interval.

## Publication bars

Use `plot_bars` for the black-edged, palette-colored, three-decimal bar style
used by the research figures. Error bars and negative values are annotated on
the correct side of their caps. Call `set_bar_limits` after all panels in a
shared-y row have been drawn.

```python
import matplotlib.pyplot as plt

from imvpy.utils import plot_bars, set_bar_limits

figure, axis = plt.subplots()
plot_bars(axis, ["Full", "Ablated"], [0.32, 0.18], yerr=[0.02, 0.03])
set_bar_limits(axis)
```

`bar_style` and `annotate_bars` expose the same choices for custom vertical bar
charts. `categorical_colors`, `sequential_cmap`, and `spectral_colors` expose
the palette independently of a particular plot.

## Panel composition

`figure_size(2, 3)` computes dimensions from the canonical panel geometry and
`label_panels(axes)` applies bold `a.`, `b.`, ... labels in row-major order.
Pass only data axes to avoid labelling colorbars. `style_axis` applies the
Cartesian frame and accepts `grid_axis="x"`, `"y"`, `"both"`, or `None`.
`apply_tight_layout` reserves room for inset colorbars and figure-level legends.

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
from imvpy.utils import FIGURE_DPI, FIGURE_FORMATS, save_figure

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
Exports use a white background and 0.04-inch padding.

The caller controls the output directory. Keep generated figures outside the
package source tree unless they are intentional documentation assets.
