import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from imvpy.utils import (
    FIGURE_DPI,
    FIGURE_FORMATS,
    plot_ablation_matrix,
    plot_imv_heatmap,
    plot_ova_boxplot,
    save_figure,
)


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
        plot_imv_heatmap(np.empty((0, 0)))
    with pytest.raises(ValueError, match="one item"):
        plot_imv_heatmap(np.eye(2), labels=["only one"])
    with pytest.raises(ValueError, match="non-empty"):
        plot_ova_boxplot([])
    with pytest.raises(ValueError, match="one item"):
        plot_ova_boxplot([[0.1, 0.2]], labels=["only one"])


def test_save_figure_writes_all_required_formats_at_800_dpi(tmp_path):
    class RecordingFigure:
        def __init__(self):
            self.calls = []

        def savefig(self, path, **kwargs):
            path.touch()
            self.calls.append((path, kwargs))

    figure = RecordingFigure()
    paths = save_figure(figure, tmp_path / "nested" / "result.png")

    assert FIGURE_DPI == 800
    assert FIGURE_FORMATS == ("png", "pdf", "svg")
    assert set(paths) == set(FIGURE_FORMATS)
    assert all(path.is_file() for path in paths.values())
    assert [path.suffix for path, _ in figure.calls] == [".png", ".pdf", ".svg"]
    assert all(options["dpi"] == 800 for _, options in figure.calls)
    assert all(options["bbox_inches"] == "tight" for _, options in figure.calls)

    with pytest.raises(ValueError, match="positive"):
        save_figure(figure, tmp_path / "invalid", dpi=0)
