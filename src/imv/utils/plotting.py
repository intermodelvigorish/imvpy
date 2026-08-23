"""Reusable, side-effect-free plotting helpers for IMV results."""

# Postponed evaluation so PEP 604 unions (``X | None``) are legal on Python 3.9,
# the oldest version this package supports.
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _axis(ax, figsize: tuple[float, float]):
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        return figure, ax, True
    return ax.figure, ax, False


def plot_imv_heatmap(matrix, *, ax=None, figsize=(6, 6), title="IMV matrix",
                     labels: Sequence[str] | None = None, fmt=".3f"):
    """Plot a square IMV matrix and return ``(figure, axis)`` or the given axis."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")
    figure, ax, created = _axis(ax, figsize)
    if labels is None and hasattr(matrix, "columns"):
        labels = [str(value) for value in matrix.columns]
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt=fmt, ax=ax,
                xticklabels=labels if labels is not None else "auto",
                yticklabels=labels if labels is not None else "auto")
    ax.set_title(title)
    return (figure, ax) if created else ax


def plot_ova_boxplot(fold_scores, *, ax=None, figsize=(6, 6), labels=None,
                     title="One-vs-rest IMV across folds"):
    """Plot fold-level one-vs-rest IMV distributions."""
    values = np.asarray(fold_scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("fold_scores must be a non-empty folds-by-classes matrix")
    figure, ax, created = _axis(ax, figsize)
    if labels is None:
        labels = [f"Outcome {index + 1}" for index in range(values.shape[1])]
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.set_title(title)
    ax.set_ylabel("IMV")
    return (figure, ax) if created else ax


def plot_ablation_matrix(matrix, **kwargs):
    """Plot a model-ablation matrix using its model names as labels."""
    kwargs.setdefault("title", "Ablation IMV matrix")
    return plot_imv_heatmap(matrix, **kwargs)
