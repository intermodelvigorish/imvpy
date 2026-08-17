from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from imv import calculate_imv


ROOT = Path(__file__).resolve().parents[1]


def test_src_layout_has_canonical_subpackages_and_no_root_python_package():
    expected = {
        ROOT / "src/imv/utils/core.py",
        ROOT / "src/imv/utils/plotting.py",
        ROOT / "src/imv/shap_imv/evaluator.py",
        ROOT / "src/imv/multi_imv/evaluator.py",
        ROOT / "src/imv/ablation_imv/evaluator.py",
    }
    assert all(path.is_file() for path in expected)
    assert not list((ROOT / "imv").glob("*.py"))


def test_examples_are_not_python_scripts_and_have_notebook_outputs():
    maintained = [
        *sorted((ROOT / "examples/shap_imv").glob("*.ipynb")),
        *sorted((ROOT / "examples/multi_imv").glob("*.ipynb")),
        *sorted((ROOT / "examples/ablation_imv").glob("*.ipynb")),
    ]
    assert len(maintained) == 9
    assert not list((ROOT / "examples").glob("**/*.py"))
    for path in maintained:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells
    for path in maintained:
        notebook = nbformat.read(path, as_version=4)
        assert any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        assert "results.pkl" not in source
        assert "BinaryIMV" in source or "MulticlassIMV" in source or "AblationIMV" in source


def test_ablation_formula_matches_independent_notebook_transcription():
    """Recompute parity live rather than trusting a stored tolerance.

    The retired ``*_parity.json`` artifacts recorded a difference measured against
    an older implementation, and the test that read them could never notice the
    code drifting away from the claim. This transcribes the original notebook
    formula here and compares it to the package on shipped predictions.
    """
    predictions = ROOT / "examples/ablation_imv/predictions"
    basic = pd.read_csv(predictions / "imdb_test_predictions_seed_42_no_attention.csv")
    enhanced = pd.read_csv(predictions / "imdb_test_predictions_seed_42.csv")
    y = basic["True Label"].to_numpy(float)
    assert np.array_equal(y, enhanced["True Label"].to_numpy(float))

    def notebook_ll(x, p, epsilon=1e-9):
        """Original form: epsilon added inside the logarithm, not clipped."""
        return np.exp(np.mean(x * np.log(p + epsilon) + (1 - x) * np.log(1 - p + epsilon)))

    def independent_root(a):
        def g(w):
            return w * np.log(w) + (1 - w) * np.log(1 - w)
        return brentq(lambda w: g(w) - np.log(a), 0.5, 1 - 1e-15, xtol=1e-16, rtol=8.9e-16)

    w0 = independent_root(notebook_ll(y, basic["Positive Probability"].to_numpy(float)))
    w1 = independent_root(notebook_ll(y, enhanced["Positive Probability"].to_numpy(float)))
    transcribed = (w1 - w0) / w0

    package = calculate_imv(
        basic["Positive Probability"].to_numpy(float),
        enhanced["Positive Probability"].to_numpy(float),
        y,
    )
    # Residual is dominated by epsilon handling: the notebook adds epsilon inside
    # the log, the package clips into [epsilon, 1-epsilon]. That is order 1e-9.
    assert package == pytest.approx(transcribed, abs=1e-8)


def test_no_stale_parity_artifacts_remain():
    """The removed artifacts must not reappear without a live check behind them."""
    assert not list((ROOT / "examples").glob("**/*parity*.json"))
