"""Reusable, side-effect-free plotting helpers for IMV results."""

# Postponed evaluation so PEP 604 unions (``X | None``) are legal on Python 3.9,
# the oldest version this package supports.
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGURE_DPI = 800
FIGURE_FORMATS = ("png", "pdf", "svg")


def _axis(ax, figsize: tuple[float, float]):
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        return figure, ax, True
    return ax.figure, ax, False


def plot_imv_heatmap(matrix, *, ax=None, figsize=(6, 6), title="IMV matrix",
                     labels: Sequence[str] | None = None, fmt=".3f"):
    """Plot a square IMV matrix as an annotated heatmap.

    Args:
        matrix (array-like): Non-empty square numeric matrix. When this is a
            pandas DataFrame, its column names become labels by default.
        ax (matplotlib.axes.Axes, optional): Existing axis. A new figure and axis
            are created when omitted.
        figsize (tuple[float, float], optional): New figure size in inches.
            Ignored when ``ax`` is supplied. Default: ``(6, 6)``.
        title (str, optional): Axis title. Default: ``"IMV matrix"``.
        labels (Sequence[str], optional): Shared row and column labels. Must have
            one item per matrix dimension. DataFrame columns are used when
            omitted.
        fmt (str, optional): seaborn annotation format. Default: ``".3f"``.

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
    figure, ax, created = _axis(ax, figsize)
    if labels is None and hasattr(matrix, "columns"):
        labels = [str(value) for value in matrix.columns]
    if labels is not None and len(labels) != data.shape[0]:
        raise ValueError("labels must contain one item per matrix dimension")
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt=fmt, ax=ax,
                xticklabels=labels if labels is not None else "auto",
                yticklabels=labels if labels is not None else "auto")
    ax.set_title(title)
    return (figure, ax) if created else ax


def plot_ova_boxplot(fold_scores, *, ax=None, figsize=(6, 6), labels=None,
                     title="One-vs-rest IMV across folds"):
    """Plot fold-level one-vs-rest IMV distributions.

    Args:
        fold_scores (array-like): Non-empty folds-by-classes numeric matrix.
        ax (matplotlib.axes.Axes, optional): Existing axis. A new figure and axis
            are created when omitted.
        figsize (tuple[float, float], optional): New figure size in inches.
            Ignored when ``ax`` is supplied. Default: ``(6, 6)``.
        labels (Sequence[str], optional): One label per class. Generic outcome
            labels are generated when omitted.
        title (str, optional): Axis title.

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
    figure, ax, created = _axis(ax, figsize)
    if labels is None:
        labels = [f"Outcome {index + 1}" for index in range(values.shape[1])]
    if len(labels) != values.shape[1]:
        raise ValueError("labels must contain one item per class")
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.set_title(title)
    ax.set_ylabel("IMV")
    return (figure, ax) if created else ax


def plot_ablation_matrix(matrix, **kwargs):
    """Plot a model-ablation matrix using its model names as labels.

    Args:
        matrix (array-like): Input forwarded to :func:`plot_imv_heatmap`.
        **kwargs (object): Additional heatmap options. The default title is
            ``"Ablation IMV matrix"``.

    Returns:
        tuple or matplotlib.axes.Axes: The return value from
        :func:`plot_imv_heatmap`.
    """
    kwargs.setdefault("title", "Ablation IMV matrix")
    return plot_imv_heatmap(matrix, **kwargs)


def save_figure(figure, destination, *, dpi=FIGURE_DPI, bbox_inches="tight",
                **savefig_kwargs):
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
        ValueError: If ``dpi`` is not positive.
    """
    if not isinstance(dpi, (int, float)) or dpi <= 0:
        raise ValueError("dpi must be a positive number")

    base = Path(destination)
    if base.suffix.lower().lstrip(".") in FIGURE_FORMATS:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    paths = {}
    for file_format in FIGURE_FORMATS:
        path = Path(f"{base}.{file_format}")
        figure.savefig(
            path,
            format=file_format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )
        paths[file_format] = path
    return paths
