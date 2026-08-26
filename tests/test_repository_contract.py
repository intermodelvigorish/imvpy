import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from imv import calculate_imv

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE_SUFFIXES = {
    ".arff",
    ".csv",
    ".data",
    ".feather",
    ".gz",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".sav",
    ".tar",
    ".tsv",
    ".xls",
    ".xlsx",
    ".zip",
}
EXCLUDED_TREES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "site",
    "venv",
}


def repository_files():
    """Yield source-controlled candidates while ignoring local build environments."""
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_TREES or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file():
            yield relative


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


def test_project_metadata_targets_the_standalone_package_repository():
    pyproject = (ROOT / "pyproject.toml").read_text()
    mkdocs = (ROOT / "mkdocs.yml").read_text()
    readme = (ROOT / "README.md").read_text()

    assert 'name = "imv"' in pyproject
    assert "github.com/intermodelvigorish/PyIMV" in pyproject
    assert "intermodelvigorish.github.io/PyIMV/" in pyproject
    for text in (pyproject, mkdocs, readme):
        assert "imv_ml_package" not in text


def test_repository_contains_no_replication_assets_or_dependencies():
    files = list(repository_files())
    assert not any(path.suffix.lower() == ".ipynb" for path in files)
    assert not any(path.parts[0] == "examples" for path in files)
    assert not any(path.parts[:2] == ("documentation", "examples") for path in files)
    assert not (ROOT / "requirements.txt").exists()

    pyproject = (ROOT / "pyproject.toml").read_text()
    for extra in ("notebooks", "examples", "examples-deep-learning"):
        assert re.search(rf"^{re.escape(extra)}\s*=", pyproject, re.MULTILINE) is None
    for replication_dependency in (
        "jupyter",
        "nbclient",
        "nbformat",
        "ucimlrepo",
        "xgboost",
        "lightgbm",
        "transformers",
        '"datasets',
    ):
        assert replication_dependency not in pyproject.lower()


def test_repository_ships_no_data_files():
    data_files = [path for path in repository_files() if path.suffix.lower() in DATA_FILE_SUFFIXES]
    assert not data_files, f"data files must not be stored in the package repository: {data_files}"


def test_pytest_warning_locations_are_relative():
    rendered = warnings.formatwarning(
        "portable warning",
        UserWarning,
        str(ROOT / "src/imv/utils/core.py"),
        1,
    )
    assert str(ROOT) not in rendered
    assert rendered.startswith("src/imv/utils/core.py:1:")


def test_package_matches_independent_transcription_of_published_formula():
    rng = np.random.default_rng(20240817)
    n = 25_000
    outcomes = rng.integers(0, 2, n).astype(float)
    basic = np.clip(rng.beta(2, 2, n) * 0.6 + outcomes * 0.2, 1e-6, 1 - 1e-6)
    enhanced = np.clip(rng.beta(2, 2, n) * 0.3 + outcomes * 0.6, 1e-6, 1 - 1e-6)

    def published_likelihood(labels, probabilities):
        log_likelihood = labels * np.log(probabilities)
        log_likelihood += (1 - labels) * np.log(1 - probabilities)
        return np.exp(np.mean(log_likelihood))

    def independent_root(likelihood):
        def equation(weight):
            return weight * np.log(weight) + (1 - weight) * np.log(1 - weight)

        return brentq(
            lambda weight: equation(weight) - np.log(likelihood),
            0.5,
            1 - 1e-15,
            xtol=1e-16,
            rtol=8.9e-16,
        )

    basic_weight = independent_root(published_likelihood(outcomes, basic))
    enhanced_weight = independent_root(published_likelihood(outcomes, enhanced))
    expected = (enhanced_weight - basic_weight) / basic_weight

    assert calculate_imv(basic, enhanced, outcomes) == pytest.approx(expected, abs=1e-12)


def test_no_stale_parity_files_remain():
    assert not [path for path in repository_files() if "parity" in path.name.lower()]
