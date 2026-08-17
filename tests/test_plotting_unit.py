import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from imv.utils import plot_ablation_matrix, plot_imv_heatmap, plot_ova_boxplot


def test_shared_heatmap_uses_dataframe_labels_and_existing_axis():
    matrix = pd.DataFrame([[0.0, 0.2], [-0.1, 0.0]], index=["a", "b"], columns=["a", "b"])
    figure, axis = plot_imv_heatmap(matrix)
    assert axis.get_title() == "IMV matrix"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["a", "b"]
    other_figure, other_axis = plt.subplots()
    assert plot_ablation_matrix(matrix, ax=other_axis) is other_axis
    plt.close(figure)
    plt.close(other_figure)


def test_shared_boxplot_and_invalid_shapes():
    figure, axis = plot_ova_boxplot(np.array([[0.1, 0.2], [0.2, 0.3]]), labels=["x", "y"])
    assert axis.get_ylabel() == "IMV"
    plt.close(figure)
    with pytest.raises(ValueError, match="square"):
        plot_imv_heatmap([[1, 2, 3]])
    with pytest.raises(ValueError, match="non-empty"):
        plot_ova_boxplot([])
