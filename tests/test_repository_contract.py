import re
from pathlib import Path

import nbformat
import numpy as np
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


EXPECTED_EXAMPLES = {
    "shap_imv": {"adult_income", "titanic", "breast_cancer", "wine_quality"},
    "multi_imv": {"nursery", "car_evaluation", "dry_bean"},
    "ablation_imv": {"imdb"},
}


def maintained_notebooks():
    return [
        path
        for family in EXPECTED_EXAMPLES
        for path in sorted((ROOT / "examples" / family).glob("*.ipynb"))
    ]


def test_every_example_family_is_present():
    for family, datasets in EXPECTED_EXAMPLES.items():
        found = {p.stem.replace(f"{family}_", "") for p in (ROOT / "examples" / family).glob("*.ipynb")}
        assert found == datasets, f"{family}: expected {datasets}, found {found}"


def test_examples_are_not_python_scripts_and_have_notebook_outputs():
    maintained = maintained_notebooks()
    assert len(maintained) == sum(len(v) for v in EXPECTED_EXAMPLES.values())
    assert not list((ROOT / "examples").glob("**/*.py"))
    for path in maintained:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells
        assert any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        assert "results.pkl" not in source
        assert "BinaryIMV" in source or "MulticlassIMV" in source or "AblationIMV" in source


def test_every_example_downloads_its_own_data_and_ships_none():
    """Each example must run from a cold clone: no dataset in the repo, and a
    download call in the notebook itself."""
    fetchers = ("fetch_ucirepo", "fetch_openml", "load_dataset")
    for path in maintained_notebooks():
        source = "\n".join(
            cell.source for cell in nbformat.read(path, as_version=4).cells
            if cell.cell_type == "code"
        )
        assert any(f in source for f in fetchers), f"{path.name} has no download call"
    # No committed dataset files anywhere under examples/.
    committed_data = [
        p for p in (ROOT / "examples").glob("**/*.csv")
        if "results" not in p.parts
    ]
    assert not committed_data, f"datasets must not be committed: {committed_data}"
    assert not list((ROOT / "examples").glob("**/*.pkl"))


def test_every_example_uses_at_least_five_seeds():
    for path in maintained_notebooks():
        source = "\n".join(
            cell.source for cell in nbformat.read(path, as_version=4).cells
            if cell.cell_type == "code"
        )
        match = re.search(r"SEEDS\s*=\s*\[([0-9,\s]+)\]", source)
        assert match, f"{path.name} does not declare SEEDS"
        seeds = [s for s in match.group(1).replace(" ", "").split(",") if s]
        assert len(seeds) >= 5, f"{path.name} uses {len(seeds)} seeds, need >= 5"


@pytest.mark.skipif(
    not (ROOT / "documentation/examples").is_dir(),
    reason="documentation/ is maintained offline and is not distributed",
)
def test_each_example_has_documentation():
    for family, datasets in EXPECTED_EXAMPLES.items():
        for dataset in datasets:
            doc = ROOT / "documentation/examples" / family / f"{dataset}.md"
            assert doc.is_file(), f"missing {doc.relative_to(ROOT)}"
            text = doc.read_text().lower()
            for section in ("download", "preprocess", "seed", "imv"):
                assert section in text, f"{doc.name} does not cover {section!r}"


def test_package_matches_independent_transcription_of_the_notebook_formula():
    """Recompute legacy parity live, from generated data rather than a data file.

    The retired ``*_parity.json`` artifacts recorded a tolerance measured against
    an older implementation, and the test that read them could never notice the
    code drifting away from the claim. Earlier this test read shipped prediction
    CSVs; those are no longer committed, so the inputs are generated here and the
    original formula is transcribed inline.
    """
    rng = np.random.default_rng(20240817)
    n = 25_000
    y = rng.integers(0, 2, n).astype(float)
    # Confident, well-calibrated-ish probabilities spanning the usable range.
    basic = np.clip(rng.beta(2, 2, n) * 0.6 + y * 0.2, 1e-6, 1 - 1e-6)
    enhanced = np.clip(rng.beta(2, 2, n) * 0.3 + y * 0.6, 1e-6, 1 - 1e-6)

    def notebook_ll(x, p, epsilon=1e-9):
        """Original form: epsilon added inside the logarithm, not clipped."""
        return np.exp(np.mean(x * np.log(p + epsilon) + (1 - x) * np.log(1 - p + epsilon)))

    def independent_root(a):
        def g(w):
            return w * np.log(w) + (1 - w) * np.log(1 - w)
        return brentq(lambda w: g(w) - np.log(a), 0.5, 1 - 1e-15, xtol=1e-16, rtol=8.9e-16)

    w0 = independent_root(notebook_ll(y, basic))
    w1 = independent_root(notebook_ll(y, enhanced))
    transcribed = (w1 - w0) / w0

    package = calculate_imv(basic, enhanced, y)
    # Residual is dominated by epsilon handling: the notebook adds epsilon inside
    # the log, the package clips into [epsilon, 1-epsilon]. That is order 1e-9.
    assert package == pytest.approx(transcribed, abs=1e-8)
    # And the legacy backend reproduces the legacy bound behaviour exactly.
    legacy = calculate_imv(basic, enhanced, y, method="lbfgsb")
    assert legacy == pytest.approx(transcribed, abs=1e-7)


def test_no_stale_parity_artifacts_remain():
    """The removed artifacts must not reappear without a live check behind them."""
    assert not list((ROOT / "examples").glob("**/*parity*.json"))
