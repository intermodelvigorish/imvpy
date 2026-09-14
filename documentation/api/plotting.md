# Plotting API

Import shared plotting utilities from `imvpy.utils`. Plotting functions apply the
IMV visual system locally; style configuration functions let custom Matplotlib
figures use the same choices.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `COLORMAP` | `"imv"` | Registered navy-to-red continuous colormap |
| `DISPLAY_DPI` | `110` | Default interactive figure resolution |
| `FIGURE_DPI` | `800` | Default raster and rasterized-artist resolution |
| `FIGURE_FORMATS` | `("png", "pdf", "svg")` | Formats emitted by `save_figure` |
| `PALETTE_COLORS` | named colors | Canonical red, cream, blue, navy, and green values |
| `PAPER_STYLE` | `dict` | Canonical publication `rcParams` |

## Style configuration

::: imvpy.utils.plotting.rc_params

::: imvpy.utils.plotting.plotting_context

::: imvpy.utils.plotting.configure_plotting

## Palette helpers

::: imvpy.utils.plotting.spectral_colors

::: imvpy.utils.plotting.categorical_colors

::: imvpy.utils.plotting.sequential_cmap

## Panel helpers

::: imvpy.utils.plotting.figure_size

::: imvpy.utils.plotting.label_panels

::: imvpy.utils.plotting.style_axis

::: imvpy.utils.plotting.apply_tight_layout

## Bar helpers

::: imvpy.utils.plotting.bar_style

::: imvpy.utils.plotting.annotate_bars

::: imvpy.utils.plotting.plot_bars

::: imvpy.utils.plotting.set_bar_limits

## Heatmap helpers

::: imvpy.utils.plotting.heatmap_style

::: imvpy.utils.plotting.style_heatmap_frame

::: imvpy.utils.plotting.style_heatmap_axes

::: imvpy.utils.plotting.add_heatmap_colorbar

## Matrix heatmap

::: imvpy.utils.plotting.plot_imv_heatmap

## One-vs-rest boxplot

::: imvpy.utils.plotting.plot_ova_boxplot

## Ablation heatmap

::: imvpy.utils.plotting.plot_ablation_matrix

## Multi-format export

::: imvpy.utils.plotting.save_figure
