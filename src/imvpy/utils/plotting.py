"""Publication-ready plotting styles and helpers for IMV results.

The default visual system mirrors the IMV research figures: a cool-to-warm
custom palette, Helvetica-compatible sans-serif typography, restrained dashed
grids, black-edged marks, and compact manuscript panel geometry. Plotting
helpers apply the style locally; :func:`configure_plotting` is available when a
caller deliberately wants to update Matplotlib's global defaults.
"""

# Postponed evaluation so PEP 604 unions (``X | None``) are legal on Python 3.9,
# the oldest version this package supports.
from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.transforms import Bbox

FIGURE_DPI = 800
FIGURE_FORMATS = ("png", "pdf", "svg")
DISPLAY_DPI = 110

# Canonical IMV palette, ordered from warm to cool for named use. The continuous
# map below runs in the opposite direction so low values read navy and high
# values read red, with cream as its neutral midpoint.
PALETTE_COLORS = {
    "red": "#E66859",
    "cream": "#FEE7BA",
    "light_blue": "#75BBD4",
    "blue": "#74ADD1",
    "steel_blue": "#416FA0",
    "navy": "#274668",
    "green": "#74D1AF",
}
COLORMAP = "imv"
COLORMAP_STOPS = (
    (0.00, PALETTE_COLORS["navy"]),
    (0.16, PALETTE_COLORS["steel_blue"]),
    (0.30, PALETTE_COLORS["blue"]),
    (0.38, PALETTE_COLORS["light_blue"]),
    (0.50, PALETTE_COLORS["cream"]),
    (1.00, PALETTE_COLORS["red"]),
)


def _register_colormaps():
    """Register stable names without overwriting another live registry entry."""
    colormap = LinearSegmentedColormap.from_list(COLORMAP, COLORMAP_STOPS)
    if COLORMAP not in mpl.colormaps:
        mpl.colormaps.register(colormap, name=COLORMAP)
    if f"{COLORMAP}_r" not in mpl.colormaps:
        mpl.colormaps.register(colormap.reversed(), name=f"{COLORMAP}_r")


_register_colormaps()

# Helvetica is canonical where available. Nimbus Sans is its metrically
# compatible Linux substitute and DejaVu Sans is Matplotlib's portable fallback.
FONT_CANDIDATES = ("Helvetica", "Nimbus Sans", "DejaVu Sans")


def _installed_font_families(candidates=FONT_CANDIDATES):
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in candidates if name in installed] or [candidates[-1]]


FONT_FAMILY = _installed_font_families()
PANEL_WIDTH = 5.4
PANEL_HEIGHT = 4.6
GRID_LINESTYLE = "--"
EDGE_COLOR = "#1f2a30"
EDGE_WIDTH = 0.6
GRID_COLOR = "#d7dcdf"
AXIS_COLOR = "#3f484d"
TEXT_COLOR = "#1f2a30"
BAR_EDGE_COLOR = "black"
ERROR_COLOR = "black"
CAPSIZE = 3
BAR_LABEL_FORMAT = ".3f"
BAR_LABEL_PADDING = 4
BAR_LIMIT_PADDING = 0.18
COLORBAR_WIDTH = 0.08
COLORBAR_PADDING = 0.04
TIGHT_LAYOUT_PAD = 0.85
TIGHT_LAYOUT_H_PAD = 0.45
TIGHT_LAYOUT_W_PAD = 0.75
EXPORT_PADDING_INCHES = 0.04

BASE_FONT_SIZE = 11
TICK_FONT_SIZE = 10
LABEL_FONT_SIZE = 11
TITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 9
MULTICLASS_ANNOTATION_FONT_SIZE = 10

PAPER_STYLE = {
    "font.family": FONT_FAMILY,
    "font.sans-serif": FONT_FAMILY,
    "mathtext.fontset": "dejavusans",
    "font.size": BASE_FONT_SIZE,
    "font.weight": "normal",
    "text.color": TEXT_COLOR,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": LABEL_FONT_SIZE,
    "axes.labelweight": "normal",
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": AXIS_COLOR,
    "axes.linewidth": EDGE_WIDTH,
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linestyle": GRID_LINESTYLE,
    "grid.linewidth": EDGE_WIDTH,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "xtick.labelcolor": TEXT_COLOR,
    "ytick.labelcolor": TEXT_COLOR,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.frameon": False,
    "figure.figsize": (PANEL_WIDTH, PANEL_HEIGHT),
    "figure.dpi": DISPLAY_DPI,
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "figure.constrained_layout.use": False,
    "savefig.facecolor": "white",
    "savefig.dpi": FIGURE_DPI,
    "savefig.pad_inches": EXPORT_PADDING_INCHES,
    "pdf.fonttype": 42,
    "image.cmap": COLORMAP,
}

TITLE_KWARGS = {
    "fontsize": TITLE_FONT_SIZE,
    "fontweight": "bold",
    "color": TEXT_COLOR,
}
LABEL_KWARGS = {
    "fontsize": LABEL_FONT_SIZE,
    "fontweight": "normal",
    "color": TEXT_COLOR,
}


def _validate_color_count(count):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("color count must be a positive integer")
    return int(count)


def spectral_colors(count):
    """Return evenly spaced categorical colors from the canonical IMV ramp.

    Args:
        count (int): Positive number of colors.

    Returns:
        numpy.ndarray: An ``(count, 4)`` array of RGBA colors.
    """
    count = _validate_color_count(count)
    return mpl.colormaps[COLORMAP](np.linspace(0.08, 0.92, count))


def categorical_colors(count):
    """Return ``count`` well-separated RGB colors from the IMV palette."""
    return [tuple(color[:3]) for color in spectral_colors(count)]


def sequential_cmap():
    """Return the continuous navy-to-red IMV colormap."""
    return mpl.colormaps[COLORMAP]


def rc_params(**overrides):
    """Return publication-style Matplotlib parameters without applying them.

    Keyword overrides are layered over the canonical defaults. Keys containing
    periods can be passed by expanding a dictionary, for example
    ``rc_params(**{"axes.grid": False})``.
    """
    settings = {
        **PAPER_STYLE,
        "axes.prop_cycle": mpl.cycler(color=spectral_colors(10)),
    }
    settings.update(overrides)
    return settings


def plotting_context(**overrides):
    """Return a scoped Matplotlib context using the IMV publication style.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> with plotting_context():
        ...     figure, axis = plt.subplots()
    """
    return mpl.rc_context(rc=rc_params(**overrides))


def configure_plotting(**overrides):
    """Apply the IMV publication style globally and return applied settings.

    Prefer :func:`plotting_context` in reusable applications because this
    function intentionally mutates Matplotlib's process-wide ``rcParams``.
    """
    settings = rc_params(**overrides)
    mpl.rcParams.update(settings)
    return settings


def figure_size(n_rows=1, n_cols=1, *, width=PANEL_WIDTH, height=PANEL_HEIGHT):
    """Return manuscript figure dimensions for a regular panel grid.

    Args:
        n_rows (int): Positive number of rows.
        n_cols (int): Positive number of columns.
        width (float): Width of each panel in inches.
        height (float): Height of each panel in inches.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("a figure needs at least one row and one column")
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("panel width and height must be finite and positive")
    return (n_cols * width, n_rows * height)


def label_panels(axes, *, start=0, fontsize=TITLE_FONT_SIZE):
    """Label data axes ``a.``, ``b.``, ... in bold row-major order.

    Pass data axes explicitly rather than ``figure.axes`` so colorbars are not
    labelled. Existing centered and right-aligned titles are cleared.
    """
    panels = np.asarray(axes, dtype=object).ravel()
    valid_start = isinstance(start, (int, np.integer)) and not isinstance(start, bool)
    if not valid_start or start < 0 or start + len(panels) > 26:
        raise ValueError("panel labels must fall between a. and z.")
    for index, axis in enumerate(panels, start=int(start)):
        axis.set_title("", loc="center")
        axis.set_title("", loc="right")
        axis.set_title(
            f"{chr(97 + index)}.",
            loc="left",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
            pad=8,
        )


def style_axis(axis, *, grid_axis="y"):
    """Apply the IMV frame, ticks, and dashed grid to a Cartesian axis.

    Args:
        axis (matplotlib.axes.Axes): Axis to style in place.
        grid_axis ({"x", "y", "both", None}): Direction in which to draw grid
            lines. Pass ``None`` to disable the grid.
    """
    if grid_axis not in {"x", "y", "both", None}:
        raise ValueError("grid_axis must be 'x', 'y', 'both', or None")
    axis.set_axisbelow(True)
    axis.set_facecolor("white")
    axis.grid(False)
    if grid_axis is not None:
        axis.grid(
            True,
            axis=grid_axis,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=EDGE_WIDTH,
        )
    for name, spine in axis.spines.items():
        spine.set_visible(name not in {"top", "right"})
        spine.set_edgecolor(AXIS_COLOR)
        spine.set_linewidth(EDGE_WIDTH)
    axis.tick_params(axis="both", colors=AXIS_COLOR, labelcolor=TEXT_COLOR)
    return axis


def bar_style(n_bars, *, colors=None):
    """Return canonical keyword arguments for bars with error bars."""
    n_bars = _validate_color_count(n_bars)
    return {
        "color": categorical_colors(n_bars) if colors is None else colors,
        "edgecolor": BAR_EDGE_COLOR,
        "linewidth": EDGE_WIDTH,
        "capsize": CAPSIZE,
        "error_kw": {
            "ecolor": ERROR_COLOR,
            "elinewidth": EDGE_WIDTH,
            "capthick": EDGE_WIDTH,
        },
    }


def annotate_bars(axis, bars, *, fontsize=ANNOTATION_FONT_SIZE):
    """Label vertical bar values beyond their error caps, including negatives."""
    if getattr(bars, "orientation", None) != "vertical":
        raise ValueError("annotate_bars expects vertical bars")
    segments = []
    if bars.errorbar is not None and bars.errorbar.has_yerr:
        segments = bars.errorbar.lines[2][-1].get_segments()
    annotations = []
    for index, rectangle in enumerate(bars.patches):
        value = rectangle.get_height()
        if not np.isfinite(value):
            continue
        positive = value >= 0
        endpoint = rectangle.get_y() + value
        if index < len(segments) and len(segments[index]):
            endpoints = segments[index][:, 1]
            endpoint = endpoints.max() if positive else endpoints.min()
        label = format(value, BAR_LABEL_FORMAT)
        if label == "-0.000":
            label = "0.000"
        annotation = axis.annotate(
            label,
            (rectangle.get_x() + rectangle.get_width() / 2, endpoint),
            textcoords="offset points",
            xytext=(0, BAR_LABEL_PADDING if positive else -BAR_LABEL_PADDING),
            ha="center",
            va="bottom" if positive else "top",
            fontsize=fontsize,
            color=TEXT_COLOR,
        )
        annotation.set_gid("bar-value")
        annotations.append(annotation)
    return annotations


def plot_bars(
    axis,
    x,
    values,
    *,
    yerr=None,
    colors=None,
    width=0.8,
    annotation_fontsize=ANNOTATION_FONT_SIZE,
):
    """Draw annotated, black-edged vertical bars in the IMV publication style.

    Use :func:`set_bar_limits` after plotting all axes in a shared-y row so
    labels and error caps receive consistent headroom.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bar values must be a non-empty one-dimensional array")
    with plotting_context():
        bars = axis.bar(
            x,
            values,
            yerr=yerr,
            width=width,
            **bar_style(len(values), colors=colors),
        )
        annotate_bars(axis, bars, fontsize=annotation_fontsize)
    style_axis(axis, grid_axis="y")
    axis.axhline(0, color=AXIS_COLOR, linewidth=EDGE_WIDTH)
    return bars


def set_bar_limits(axes, *, padding=BAR_LIMIT_PADDING):
    """Add shared headroom beyond bar/error extents without resizing axes."""
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("bar-limit padding must be finite and positive")
    visited = set()
    for axis in np.asarray(axes, dtype=object).ravel():
        if axis in visited:
            continue
        group = axis.get_shared_y_axes().get_siblings(axis)
        visited.update(group)
        bounds = np.array([sibling.dataLim.intervaly for sibling in group])
        bounds = bounds[np.isfinite(bounds).all(axis=1)]
        if not len(bounds):
            continue
        low, high = min(0, bounds[:, 0].min()), max(0, bounds[:, 1].max())
        tolerance = 8 * np.finfo(float).eps * max(abs(low), abs(high))
        if low < 0 and abs(low) <= tolerance:
            low = 0.0
        margin = padding * (high - low if high > low else 1.0)
        axis.set_ylim(low - margin if low < 0 else 0, high + margin)


def heatmap_style(n_classes, *, annotation_fontsize=MULTICLASS_ANNOTATION_FONT_SIZE):
    """Return canonical seaborn options for an annotated IMV heatmap."""
    n_classes = _validate_color_count(n_classes)
    return {
        "cmap": sequential_cmap(),
        "annot": True,
        "fmt": ".2f" if n_classes > 5 else ".3f",
        "annot_kws": {"fontsize": annotation_fontsize},
        "linewidths": EDGE_WIDTH,
        "linecolor": BAR_EDGE_COLOR,
        "square": True,
    }


def style_heatmap_frame(axis):
    """Show a thin black frame around a heatmap and remove Cartesian grids."""
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BAR_EDGE_COLOR)
        spine.set_linewidth(EDGE_WIDTH)
    return axis


def style_heatmap_axes(axis, *, xlabels=None, ylabels=None):
    """Frame a heatmap and repeat its class labels on opposing edges."""
    if xlabels is not None:
        axis.set_xticks(axis.get_xticks(), labels=xlabels)
    if ylabels is not None:
        axis.set_yticks(axis.get_yticks(), labels=ylabels)
    style_heatmap_frame(axis)
    axis.tick_params(
        axis="both",
        which="major",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=True,
        labelbottom=True,
        labelleft=True,
        labelright=True,
        direction="out",
        length=3,
        width=EDGE_WIDTH,
        color=BAR_EDGE_COLOR,
        labelsize=TICK_FONT_SIZE,
    )
    for tick in axis.xaxis.get_major_ticks():
        for label, alignment in ((tick.label1, "right"), (tick.label2, "left")):
            label.set_rotation(45)
            label.set_rotation_mode("anchor")
            label.set_horizontalalignment(alignment)
    for tick in axis.yaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label2.set_rotation(0)
    return axis


def add_heatmap_colorbar(
    mappable,
    axis,
    label,
    *,
    width=COLORBAR_WIDTH,
    pad=COLORBAR_PADDING,
):
    """Add a full-height vector colorbar without resizing sibling panels."""
    if not np.isfinite(width) or width <= 0 or not np.isfinite(pad) or pad < 0:
        raise ValueError("colorbar width must be positive and padding nonnegative")
    colorbar_axis = axis.inset_axes([1 + pad, 0, width, 1])

    def locate_colorbar(child, renderer):
        del child
        bounds = axis.get_window_extent(renderer)
        right_labels = [
            tick.label2 for tick in axis.yaxis.get_major_ticks() if tick.label2.get_visible()
        ]
        decorations = right_labels + axis.get_xticklabels() if right_labels else []
        right = max(
            [bounds.x1]
            + [item.get_window_extent(renderer).x1 for item in decorations if item.get_visible()]
        )
        return Bbox.from_bounds(
            right + pad * bounds.width,
            bounds.y0,
            width * bounds.width,
            bounds.height,
        ).transformed(axis.figure.transFigure.inverted())

    colorbar_axis.set_axes_locator(locate_colorbar)
    colorbar = axis.figure.colorbar(mappable, cax=colorbar_axis, orientation="vertical")
    colorbar.set_label(label, **LABEL_KWARGS)
    colorbar.ax.tick_params(
        labelsize=TICK_FONT_SIZE,
        width=EDGE_WIDTH,
        length=3,
        colors=AXIS_COLOR,
    )
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor(BAR_EDGE_COLOR)
    colorbar.outline.set_linewidth(EDGE_WIDTH)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    return colorbar


def _style_existing_colorbar(colorbar):
    colorbar.ax.tick_params(
        labelsize=TICK_FONT_SIZE,
        width=EDGE_WIDTH,
        length=3,
        colors=AXIS_COLOR,
    )
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor(BAR_EDGE_COLOR)
    colorbar.outline.set_linewidth(EDGE_WIDTH)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")


def _contrast_heatmap_annotations(axis, data):
    mesh = axis.collections[0]
    for annotation in axis.texts:
        column, row = (int(np.floor(value)) for value in annotation.get_position())
        if row >= data.shape[0] or column >= data.shape[1]:
            continue
        value = data[row, column]
        if not np.isfinite(value):
            continue
        rgb = mesh.cmap(mesh.norm(value))[:3]
        luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
        annotation.set_color("white" if luminance < 0.48 else TEXT_COLOR)


def _centered_norm(data, center):
    if isinstance(center, bool) or not isinstance(center, Real) or not np.isfinite(center):
        raise ValueError("heatmap center must be a finite number or None")
    finite = data[np.isfinite(data)]
    distance = float(np.max(np.abs(finite - float(center)))) if finite.size else 1.0
    if distance == 0:
        distance = 1.0
    return TwoSlopeNorm(
        vmin=float(center) - distance,
        vcenter=float(center),
        vmax=float(center) + distance,
    )


def _axis(ax, figsize):
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        return figure, ax, True
    return ax.figure, ax, False


def plot_imv_heatmap(
    matrix,
    *,
    ax=None,
    figsize=(6, 6),
    title="IMV matrix",
    labels: Sequence[str] | None = None,
    fmt=".3f",
    cmap=COLORMAP,
    center=None,
    colorbar_label="IMV",
):
    """Plot a square IMV matrix using the canonical publication heatmap.

    Args:
        matrix (array-like): Non-empty square numeric matrix. When this is a
            pandas DataFrame, its column names become labels by default.
        ax (matplotlib.axes.Axes, optional): Existing axis. A new figure and axis
            are created when omitted.
        figsize (tuple[float, float], optional): New figure size in inches.
            Ignored when ``ax`` is supplied. Default: ``(6, 6)``.
        title (str, optional): Bold, left-aligned axis title.
        labels (Sequence[str], optional): Shared row and column labels. Must have
            one item per matrix dimension. DataFrame columns are used when
            omitted.
        fmt (str, optional): Seaborn annotation format. Default: ``".3f"``.
        cmap (str or matplotlib.colors.Colormap, optional): Color map. Defaults
            to the navy-to-red ``"imv"`` map.
        center (float, optional): Value placed at the neutral cream midpoint.
            Use ``0`` for directional matrices containing negative values.
        colorbar_label (str, optional): Colorbar label. Default: ``"IMV"``.

    Returns:
        tuple or matplotlib.axes.Axes: ``(figure, axis)`` for a newly created
        axis, otherwise the supplied axis.

    Raises:
        ValueError: If the matrix is empty or not square, or label count does not
            match its dimension.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[0] != data.shape[1]:
        raise ValueError("matrix must be a non-empty square two-dimensional array")
    if labels is None and hasattr(matrix, "columns"):
        labels = [str(value) for value in matrix.columns]
    if labels is not None and len(labels) != data.shape[0]:
        raise ValueError("labels must contain one item per matrix dimension")

    with plotting_context():
        figure, ax, created = _axis(ax, figsize)
        options = {
            **heatmap_style(data.shape[0]),
            "fmt": fmt,
            "cmap": cmap,
            "xticklabels": labels if labels is not None else "auto",
            "yticklabels": labels if labels is not None else "auto",
            "cbar_kws": {"label": colorbar_label},
        }
        if center is not None:
            options["norm"] = _centered_norm(data, center)
        sns.heatmap(data, ax=ax, **options)
        ax.set_title(title, **TITLE_KWARGS)
        style_heatmap_frame(ax)
        ax.tick_params(axis="both", length=3, width=EDGE_WIDTH, colors=AXIS_COLOR)
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), rotation=0)
        _contrast_heatmap_annotations(ax, data)
        if ax.collections[0].colorbar is not None:
            _style_existing_colorbar(ax.collections[0].colorbar)
    return (figure, ax) if created else ax


def plot_ova_boxplot(
    fold_scores,
    *,
    ax=None,
    figsize=(6, 6),
    labels=None,
    title="One-vs-rest IMV across folds",
    ylabel="IMV",
):
    """Plot fold-level one-vs-rest IMV distributions in publication style.

    Args:
        fold_scores (array-like): Non-empty folds-by-classes numeric matrix.
        ax (matplotlib.axes.Axes, optional): Existing axis. A new figure and axis
            are created when omitted.
        figsize (tuple[float, float], optional): New figure size in inches.
            Ignored when ``ax`` is supplied. Default: ``(6, 6)``.
        labels (Sequence[str], optional): One label per class. Generic outcome
            labels are generated when omitted.
        title (str, optional): Bold, left-aligned axis title.
        ylabel (str, optional): Vertical axis label. Default: ``"IMV"``.

    Returns:
        tuple or matplotlib.axes.Axes: ``(figure, axis)`` for a newly created
        axis, otherwise the supplied axis.

    Raises:
        ValueError: If scores are not a non-empty two-dimensional matrix or the
            label count does not match the class count.
    """
    values = np.asarray(fold_scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("fold_scores must be a non-empty folds-by-classes matrix")
    if labels is None:
        labels = [f"Outcome {index + 1}" for index in range(values.shape[1])]
    if len(labels) != values.shape[1]:
        raise ValueError("labels must contain one item per class")

    with plotting_context():
        figure, ax, created = _axis(ax, figsize)
        artists = ax.boxplot(values, patch_artist=True, widths=0.65)
        for patch, color in zip(artists["boxes"], spectral_colors(values.shape[1])):
            patch.set_facecolor(color)
            patch.set_edgecolor(BAR_EDGE_COLOR)
            patch.set_linewidth(EDGE_WIDTH)
        for name in ("whiskers", "caps", "medians"):
            for artist in artists[name]:
                artist.set_color(BAR_EDGE_COLOR)
                artist.set_linewidth(EDGE_WIDTH)
        for flier in artists["fliers"]:
            flier.set_markeredgecolor(BAR_EDGE_COLOR)
            flier.set_markersize(3)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        ax.set_title(title, **TITLE_KWARGS)
        ax.set_ylabel(ylabel, **LABEL_KWARGS)
        style_axis(ax, grid_axis="y")
        ax.axhline(0, color=AXIS_COLOR, linewidth=EDGE_WIDTH)
        if any(len(str(label)) > 8 for label in labels):
            plt.setp(
                ax.get_xticklabels(),
                rotation=42,
                ha="right",
                rotation_mode="anchor",
            )
    return (figure, ax) if created else ax


def plot_ablation_matrix(matrix, **kwargs):
    """Plot a directional model-ablation matrix in the IMV visual style.

    Args:
        matrix (array-like): Input forwarded to :func:`plot_imv_heatmap`.
        **kwargs (object): Additional heatmap options. By default zero is placed
            at the palette's neutral midpoint and the colorbar is labelled
            ``"Directional IMV"``.

    Returns:
        tuple or matplotlib.axes.Axes: The return value from
        :func:`plot_imv_heatmap`.
    """
    kwargs.setdefault("title", "Ablation IMV matrix")
    kwargs.setdefault("center", 0.0)
    kwargs.setdefault("colorbar_label", "Directional IMV")
    return plot_imv_heatmap(matrix, **kwargs)


def apply_tight_layout(figure):
    """Fit panels, inset colorbars, and bottom figure legends inside a figure."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legends = [
        legend for legend in figure.legends if legend.get_visible() and legend.get_in_layout()
    ]
    bottom = 0.0
    if legends:
        bottom = max(
            legend.get_window_extent(renderer).transformed(figure.transFigure.inverted()).y1
            for legend in legends
        )
        bottom += TIGHT_LAYOUT_PAD * mpl.rcParams["font.size"] / 72 / figure.get_figheight()
    figure.set_layout_engine(
        "tight",
        pad=TIGHT_LAYOUT_PAD,
        h_pad=TIGHT_LAYOUT_H_PAD,
        w_pad=TIGHT_LAYOUT_W_PAD,
        rect=(0, bottom, 1, 1),
    )
    figure.canvas.draw()
    return figure


def save_figure(
    figure,
    destination,
    *,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
    **savefig_kwargs,
):
    """Save a figure as 800-DPI PNG, PDF, and SVG files.

    ``destination`` may be a bare basename or end in one of the supported
    extensions; in either case all three sibling files are written. The DPI is
    also passed to vector backends so any rasterized artists use the same output
    resolution.

    Args:
        figure (matplotlib.figure.Figure): Figure exposing ``savefig``.
        destination (path-like): Output basename, optionally ending in ``.png``,
            ``.pdf``, or ``.svg``. Parent directories are created.
        dpi (int or float, optional): Positive output resolution. Default: 800.
        bbox_inches (str, optional): Matplotlib bounding-box mode. Default:
            ``"tight"``.
        **savefig_kwargs (object): Additional keyword arguments forwarded to every
            ``figure.savefig`` call.

    Returns:
        dict[str, pathlib.Path]: Paths keyed by ``"png"``, ``"pdf"``, and
        ``"svg"``.

    Raises:
        ValueError: If ``dpi`` is not a finite positive number.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, Real) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a finite positive number")

    base = Path(destination).expanduser()
    if base.suffix.lower().lstrip(".") in FIGURE_FORMATS:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "facecolor": "white",
        "pad_inches": EXPORT_PADDING_INCHES,
        **savefig_kwargs,
    }
    paths = {}
    with mpl.rc_context(PAPER_STYLE):
        for file_format in FIGURE_FORMATS:
            path = Path(f"{base}.{file_format}")
            figure.savefig(
                path,
                format=file_format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                **options,
            )
            paths[file_format] = path
    return paths
