import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from imvpy import BinaryIMV, MulticlassIMV
from imvpy.utils import (
    COLORMAP,
    DISPLAY_DPI,
    FIGURE_DPI,
    FIGURE_FORMATS,
    PALETTE_COLORS,
    PAPER_STYLE,
    add_heatmap_colorbar,
    annotate_bars,
    apply_tight_layout,
    bar_style,
    categorical_colors,
    configure_plotting,
    figure_size,
    heatmap_style,
    label_panels,
    plot_ablation_matrix,
    plot_bars,
    plot_imv_heatmap,
    plot_ova_boxplot,
    plotting_context,
    rc_params,
    save_figure,
    sequential_cmap,
    set_bar_limits,
    spectral_colors,
    style_axis,
    style_heatmap_axes,
    style_heatmap_frame,
)


def test_publication_palette_and_style_are_canonical_and_scoped():
    assert COLORMAP == "imv"
    assert DISPLAY_DPI == 110
    assert PALETTE_COLORS["navy"] == "#274668"
    assert PALETTE_COLORS["cream"] == "#FEE7BA"
    assert PALETTE_COLORS["red"] == "#E66859"
    assert mpl.colormaps[COLORMAP].name == COLORMAP
    np.testing.assert_allclose(
        mpl.colors.to_rgb(mpl.colormaps[COLORMAP](0.0)),
        mpl.colors.to_rgb(PALETTE_COLORS["navy"]),
        atol=1 / 255,
    )
    np.testing.assert_allclose(
        mpl.colors.to_rgb(mpl.colormaps[COLORMAP](1.0)),
        mpl.colors.to_rgb(PALETTE_COLORS["red"]),
        atol=1 / 255,
    )

    with mpl.rc_context({"axes.titlelocation": "center", "grid.linestyle": ":"}):
        with plotting_context():
            assert mpl.rcParams["axes.titlelocation"] == "left"
            assert mpl.rcParams["grid.linestyle"] == "--"
            assert mpl.rcParams["image.cmap"] == COLORMAP
        assert mpl.rcParams["axes.titlelocation"] == "center"
        assert mpl.rcParams["grid.linestyle"] == ":"

        settings = configure_plotting(**{"axes.grid": False})
        assert settings["axes.grid"] is False
        assert mpl.rcParams["axes.grid"] is False


def test_rc_params_palette_and_panel_geometry():
    settings = rc_params()
    assert settings["font.family"] == PAPER_STYLE["font.family"]
    assert settings["pdf.fonttype"] == 42
    assert settings["figure.figsize"] == (5.4, 4.6)
    np.testing.assert_allclose(settings["axes.prop_cycle"].by_key()["color"], spectral_colors(10))
    assert figure_size(2, 3) == pytest.approx((16.2, 9.2))
    assert figure_size(2, 3, width=4, height=2) == (12, 4)
    with pytest.raises(ValueError, match="one row"):
        figure_size(0, 1)
    with pytest.raises(ValueError, match="finite"):
        figure_size(width=np.nan)

    assert sequential_cmap().name == COLORMAP
    assert len(categorical_colors(4)) == 4
    assert heatmap_style(5)["fmt"] == ".3f"
    assert heatmap_style(6)["fmt"] == ".2f"


def test_heatmap_uses_dataframe_labels_palette_and_publication_frame():
    matrix = pd.DataFrame(
        [[0.0, 0.2], [-0.1, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    figure, axis = plot_imv_heatmap(matrix)
    assert axis.get_title(loc="left") == "IMV matrix"
    assert axis.get_title(loc="center") == ""
    assert axis._left_title.get_fontweight() == "bold"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["a", "b"]
    assert axis.collections[0].cmap.name == COLORMAP
    assert axis.collections[0].colorbar.ax.get_ylabel() == "IMV"
    assert not axis.collections[0].colorbar.solids.get_rasterized()
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert all(text.get_color() in {"white", "#1f2a30"} for text in axis.texts)

    other_figure, other_axis = plt.subplots()
    assert plot_ablation_matrix(matrix, ax=other_axis) is other_axis
    assert other_axis.get_title(loc="left") == "Ablation IMV matrix"
    assert other_axis.collections[0].colorbar.ax.get_ylabel() == "Directional IMV"
    assert other_axis.collections[0].cmap.name == COLORMAP
    assert (
        mpl.colors.to_hex(other_axis.collections[0].cmap(other_axis.collections[0].norm(0)))
        == PALETTE_COLORS["cream"].lower()
    )
    plt.close(figure)
    plt.close(other_figure)


def test_boxplot_uses_palette_black_edges_and_dashed_grid():
    figure, axis = plot_ova_boxplot(
        np.array([[0.1, 0.2], [0.2, 0.3]]),
        labels=["first class", "second class"],
    )
    boxes = axis.patches
    assert len(boxes) == 2
    np.testing.assert_allclose([box.get_facecolor() for box in boxes], spectral_colors(2))
    for box in boxes:
        np.testing.assert_array_equal(box.get_edgecolor(), [0, 0, 0, 1])
    assert axis.get_ylabel() == "IMV"
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert all(line.get_linestyle() == "--" for line in axis.get_ygridlines())
    assert all(label.get_rotation() == 42 for label in axis.get_xticklabels())
    plt.close(figure)


def test_bar_and_panel_primitives_match_research_figures():
    figure, axes = plt.subplots(1, 2, sharey=True)
    bars = plot_bars(axes[0], ["A", "B"], [0.25, -0.125], yerr=[0.02, 0.03])
    plot_bars(axes[1], ["C"], [0.5], yerr=[0.05])
    set_bar_limits(axes)
    label_panels(axes)

    np.testing.assert_allclose([bar.get_facecolor() for bar in bars], spectral_colors(2))
    assert [text.get_text() for text in axes[0].texts] == ["0.250", "-0.125"]
    assert axes[0].get_title(loc="left") == "a."
    assert axes[1].get_title(loc="left") == "b."
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert axes[0].get_ylim()[0] < -0.155
    assert axes[0].get_ylim()[1] > 0.55
    plt.close(figure)

    with pytest.raises(ValueError, match="a. and z."):
        label_panels(plt.subplots()[1], start=26)
    plt.close("all")


def test_low_level_heatmap_colorbar_and_layout_preserve_geometry():
    with plotting_context(**{"figure.autolayout": False}):
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        mesh = axes[0].pcolormesh([[0.0, 0.5], [0.5, 1.0]], cmap=COLORMAP)
        axes[0].set_xticks([0.5, 1.5], ["basic", "enhanced"])
        axes[0].set_yticks([0.5, 1.5], ["basic", "enhanced"])
        before = axes[0].get_position().bounds

        assert style_heatmap_frame(axes[0]) is axes[0]
        assert style_heatmap_axes(axes[0]) is axes[0]
        colorbar = add_heatmap_colorbar(mesh, axes[0], "Directional IMV")
        figure.canvas.draw()
        np.testing.assert_allclose(axes[0].get_position().bounds, before)
        apply_tight_layout(figure)
        figure.canvas.draw()

    assert colorbar.ax.get_ylabel() == "Directional IMV"
    assert not colorbar.solids.get_rasterized()
    assert all(spine.get_visible() for spine in axes[0].spines.values())
    assert all(tick.label2.get_visible() for tick in axes[0].xaxis.get_major_ticks())
    assert all(tick.label2.get_visible() for tick in axes[0].yaxis.get_major_ticks())
    assert colorbar.ax.get_position().height == pytest.approx(axes[0].get_position().height)
    assert figure.get_tight_layout()
    plt.close(figure)


def test_style_and_bar_primitive_validation():
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="grid_axis"):
        style_axis(axis, grid_axis="diagonal")
    with pytest.raises(ValueError, match="positive integer"):
        bar_style(0)
    horizontal = axis.barh([0], [0.2])
    with pytest.raises(ValueError, match="vertical"):
        annotate_bars(axis, horizontal)
    mesh = axis.pcolormesh([[0, 1]])
    for kwargs in ({"width": 0}, {"width": np.nan}, {"pad": -1}):
        with pytest.raises(ValueError, match="colorbar width"):
            add_heatmap_colorbar(mesh, axis, "IMV", **kwargs)
    plt.close(figure)


def test_evaluator_plotting_methods_use_shared_style():
    multiclass = MulticlassIMV.__new__(MulticlassIMV)
    heat_figure, heat_axis = multiclass.multinomial_IMV_heatmap(np.eye(2))
    box_figure, box_axis = multiclass.multinomial_IMV_boxplot([[0.1, 0.2], [0.2, 0.3]])
    assert heat_axis.collections[0].cmap.name == COLORMAP
    assert heat_axis.get_title(loc="left") == "IMV Confusion Matrix"
    assert box_axis.patches

    binary = BinaryIMV.__new__(BinaryIMV)
    binary.optional_explanatory_variables = ["age", "income"]
    binary.calculate_imvshapley_value = lambda variable: {"age": 0.2, "income": 0.1}[variable]
    bar_figure, bar_axis = binary.evaluate_imvshapley()
    assert len(bar_axis.patches) == 2
    np.testing.assert_allclose(
        [patch.get_facecolor() for patch in bar_axis.patches],
        spectral_colors(2),
    )
    assert not bar_axis.spines["top"].get_visible()
    assert all(line.get_linestyle() == "--" for line in bar_axis.get_xgridlines())

    binary.all_combinations_imv = {
        ("age",): (0.2, [0.1, 0.2, 0.3]),
        ("income",): (0.1, [0.0, 0.1, 0.2]),
    }
    violin_figure, violin_axis = (
        binary.plot_single_var_combinations_layered_violin_centralized_zero()
    )
    assert violin_axis.get_title(loc="left") == "Single Variable Model vs Null Model"
    assert not violin_axis.spines["top"].get_visible()

    for figure in (heat_figure, box_figure, bar_figure, violin_figure):
        plt.close(figure)


def test_invalid_plot_inputs_are_rejected():
    with pytest.raises(ValueError, match="square"):
        plot_imv_heatmap([[1, 2, 3]])
    with pytest.raises(ValueError, match="non-empty"):
        plot_imv_heatmap(np.empty((0, 0)))
    with pytest.raises(ValueError, match="one item"):
        plot_imv_heatmap(np.eye(2), labels=["only one"])
    with pytest.raises(ValueError, match="finite number"):
        plot_imv_heatmap(np.eye(2), center=np.nan)
    with pytest.raises(ValueError, match="non-empty"):
        plot_ova_boxplot([])
    with pytest.raises(ValueError, match="one item"):
        plot_ova_boxplot([[0.1, 0.2]], labels=["only one"])
    with pytest.raises(ValueError, match="positive integer"):
        spectral_colors(0)


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
    assert all(options["facecolor"] == "white" for _, options in figure.calls)
    assert all(options["pad_inches"] == 0.04 for _, options in figure.calls)

    for invalid in (0, np.nan, True):
        with pytest.raises(ValueError, match="finite positive"):
            save_figure(figure, tmp_path / "invalid", dpi=invalid)


def test_real_heatmap_exports_remain_vector_in_pdf_and_svg(tmp_path):
    figure, _ = plot_ablation_matrix([[0.0, 0.2], [-0.2, 0.0]])
    paths = save_figure(figure, tmp_path / "directional", dpi=72)

    assert paths["png"].read_bytes().startswith(b"\x89PNG")
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    assert b"/Subtype /Image" not in paths["pdf"].read_bytes()
    assert "<svg" in paths["svg"].read_text()
    assert "<image" not in paths["svg"].read_text()
    plt.close(figure)
