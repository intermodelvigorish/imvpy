import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import brentq

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib

from imvpy import calculate_imv

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

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
        ROOT / "src/imvpy/utils/core.py",
        ROOT / "src/imvpy/utils/plotting.py",
        ROOT / "src/imvpy/shap_imv/evaluator.py",
        ROOT / "src/imvpy/multi_imv/evaluator.py",
        ROOT / "src/imvpy/ablation_imv/evaluator.py",
    }
    assert all(path.is_file() for path in expected)
    assert not (ROOT / "src/imv").exists()
    assert not list((ROOT / "imvpy").glob("*.py"))


def test_project_metadata_targets_the_standalone_package_repository():
    project = PYPROJECT["project"]
    mkdocs = (ROOT / "mkdocs.yml").read_text()
    readme = (ROOT / "README.md").read_text()

    assert project["name"] == "imvpy"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["readme"]["content-type"] == "text/markdown"
    assert project["requires-python"] == ">=3.9"
    assert project["urls"]["Repository"] == "https://github.com/intermodelvigorish/imvpy"
    assert project["urls"]["Documentation"] == "https://intermodelvigorish.github.io/imvpy/"
    assert "github.com/intermodelvigorish/imvpy" in mkdocs
    assert "github.com/intermodelvigorish/imvpy" in readme
    assert "python -m pip install imvpy" in readme
    assert not (ROOT / "setup.py").exists()

    expected_python_classifiers = {
        f"Programming Language :: Python :: 3.{minor}" for minor in range(9, 15)
    }
    assert expected_python_classifiers <= set(project["classifiers"])
    assert project["keywords"]


def test_release_support_files_are_present_and_versioned():
    for filename in (
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "MANIFEST.in",
        "RELEASING.md",
        "SECURITY.md",
    ):
        assert (ROOT / filename).is_file()

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["version"] == PYPROJECT["project"]["version"]
    assert citation["license"] == "MIT"
    assert citation["preferred-citation"]["doi"] == "10.1371/journal.pone.0316491"


def test_repository_contains_no_replication_assets():
    files = list(repository_files())
    assert not any(path.suffix.lower() == ".ipynb" for path in files)

    declared_extras = set(PYPROJECT["project"]["optional-dependencies"])
    assert declared_extras == {
        "progress",
        "deep-learning",
        "test",
        "docs",
        "release",
        "dev",
    }


def test_repository_ships_no_data_files():
    data_files = [path for path in repository_files() if path.suffix.lower() in DATA_FILE_SUFFIXES]
    assert not data_files, f"data files must not be stored in the package repository: {data_files}"


def test_pytest_warning_locations_are_relative():
    rendered = warnings.formatwarning(
        "portable warning",
        UserWarning,
        str(ROOT / "src/imvpy/utils/core.py"),
        1,
    )
    assert str(ROOT) not in rendered
    assert rendered.startswith("src/imvpy/utils/core.py:1:")


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


def test_publishing_workflows_use_oidc_and_separate_builds():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    versions = ci["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert versions == ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
    assert ci["jobs"]["build"]["needs"] == ["lint", "test", "deep-learning"]
    deep_learning = ci["jobs"]["deep-learning"]
    deep_learning_text = yaml.safe_dump(deep_learning)
    assert 'python-version: "3.14"' in (ROOT / ".github/workflows/ci.yml").read_text()
    assert "https://download.pytorch.org/whl/cpu" in deep_learning_text
    assert "tests/test_ablation_torch.py" in deep_learning_text

    workflows = {
        "publish-test.yml": ("testpypi", "https://test.pypi.org/legacy/"),
        "publish.yml": ("pypi", None),
    }
    for filename, (environment, repository_url) in workflows.items():
        text = (ROOT / ".github/workflows" / filename).read_text()
        workflow = yaml.safe_load(text)
        assert set(workflow["jobs"]) == {"build", "publish"}
        publish = workflow["jobs"]["publish"]
        assert publish["needs"] == "build"
        assert publish["environment"]["name"] == environment
        assert publish["permissions"] == {"id-token": "write"}
        assert "pypa/gh-action-pypi-publish@release/v1" in text
        assert "PYPI_TOKEN" not in text
        assert "password:" not in text
        if repository_url is not None:
            assert repository_url in text

    production = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "release tag {actual!r} must equal {expected!r}" in production


def test_documentation_workflow_builds_strictly_and_deploys_with_oidc():
    text = (ROOT / ".github/workflows/docs.yml").read_text()
    workflow = yaml.safe_load(text)

    assert set(workflow["jobs"]) == {"build", "deploy"}
    assert workflow["jobs"]["deploy"]["needs"] == "build"
    assert workflow["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    assert workflow["jobs"]["deploy"]["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }
    assert "mkdocs build --strict --quiet" in text
    assert "actions/configure-pages@v5" in text
    assert "actions/upload-pages-artifact@v4" in text
    assert "actions/deploy-pages@v4" in text
