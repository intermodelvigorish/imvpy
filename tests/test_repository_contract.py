import ast
import re
import warnings
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
    "ablation_imv": {"har", "imdb", "mnist"},
}

EXPECTED_DATA_FETCHES = {
    "shap_imv_adult_income.ipynb": ("fetch_ucirepo", "id", 2),
    "shap_imv_titanic.ipynb": ("fetch_openml", 0, "titanic"),
    "shap_imv_breast_cancer.ipynb": ("fetch_ucirepo", "id", 17),
    "shap_imv_wine_quality.ipynb": ("fetch_ucirepo", "id", 186),
    "multi_imv_nursery.ipynb": ("fetch_ucirepo", "id", 76),
    "multi_imv_car_evaluation.ipynb": ("fetch_ucirepo", "id", 19),
    "multi_imv_dry_bean.ipynb": ("fetch_ucirepo", "id", 602),
    "ablation_imv_imdb.ipynb": ("load_dataset", 0, "stanfordnlp/imdb"),
    "ablation_imv_mnist.ipynb": ("fetch_openml", 0, "mnist_784"),
    "ablation_imv_har.ipynb": (
        "urlretrieve",
        0,
        "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip",
    ),
}

EXPECTED_PACKAGE_USAGE = {
    "shap_imv": (
        "BinaryIMV",
        {"run_evaluation", "calculate_imvshapley_value"},
    ),
    "multi_imv": (
        "MulticlassIMV",
        {"k_fold_one_vs_all", "k_fold_imv_matrix"},
    ),
    "ablation_imv": (
        "AblationIMV",
        {"train_and_evaluate", "calculate_imv_matrix", "average_imv_matrices"},
    ),
}

EXPECTED_FIGURE_EXPORTS = {
    "shap_imv_adult_income.ipynb": 1,
    "shap_imv_titanic.ipynb": 1,
    "shap_imv_breast_cancer.ipynb": 1,
    "shap_imv_wine_quality.ipynb": 1,
    "multi_imv_nursery.ipynb": 1,
    "multi_imv_car_evaluation.ipynb": 1,
    "multi_imv_dry_bean.ipynb": 1,
    "ablation_imv_imdb.ipynb": 2,
    "ablation_imv_mnist.ipynb": 2,
    "ablation_imv_har.ipynb": 2,
}

TABULAR_MODELS = {"lightgbm", "logistic_regression", "xgboost"}
EXPECTED_SEEDS = list(range(42, 52))
EXPECTED_SEEDED_MODELS = {
    "shap_imv_adult_income.ipynb": TABULAR_MODELS,
    "shap_imv_titanic.ipynb": TABULAR_MODELS,
    "shap_imv_breast_cancer.ipynb": TABULAR_MODELS,
    "shap_imv_wine_quality.ipynb": TABULAR_MODELS,
    "multi_imv_nursery.ipynb": TABULAR_MODELS,
    "multi_imv_car_evaluation.ipynb": TABULAR_MODELS,
    "multi_imv_dry_bean.ipynb": TABULAR_MODELS,
    "ablation_imv_imdb.ipynb": {
        "Original", "3Layers", "NoAttention", "NoFFN", "NoNorm",
    },
    "ablation_imv_mnist.ipynb": {
        "FullCNN", "NoConv2", "NoHidden", "NoDropout", "Linear",
    },
    "ablation_imv_har.ipynb": {
        "FullBiGRU", "UniGRU", "OneLayer", "NoAttention", "MeanPoolMLP",
    },
}

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

ABSOLUTE_PATH_PATTERNS = {
    "POSIX": re.compile(r"(?<![A-Za-z0-9:/.<~])/(?!/)(?:[^/\s<>'\"]+/)+[^/\s<>'\"]*"),
    "Windows drive": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "Windows UNC": re.compile(r"(?<!\\)\\\\[^\\/\s]+[\\/]"),
    "file URI": re.compile(r"\bfile:" r"//", re.IGNORECASE),
}


def maintained_notebooks():
    return [
        path
        for family in EXPECTED_EXAMPLES
        for path in sorted((ROOT / "examples" / family).glob("*.ipynb"))
    ]


def notebook_python(path):
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    # IPython magics are not Python AST nodes and are irrelevant to contracts.
    python_source = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    return source, ast.parse(python_source, filename=path.name)


def notebook_output_text(path):
    """Flatten textual stored outputs without scanning encoded image payloads."""
    notebook = nbformat.read(path, as_version=4)
    parts = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if "text" in output:
                value = output["text"]
                parts.append("".join(value) if isinstance(value, list) else str(value))
            for mime_type, value in output.get("data", {}).items():
                if not mime_type.startswith("text/") and mime_type != "application/json":
                    continue
                parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def notebook_markdown_text(path):
    """Flatten Markdown cells because their source is rendered directly to users."""
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )


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
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        assert [cell.execution_count for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        ), f"{path.name} is not fully and sequentially executed"
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        assert not errors, f"{path.name} contains stored execution errors"
        assert any(cell.get("outputs") for cell in code_cells)
        source = "\n".join(cell.source for cell in code_cells)
        assert "results.pkl" not in source
        assert "BinaryIMV" in source or "MulticlassIMV" in source or "AblationIMV" in source
        if path.parent.name == "ablation_imv":
            assert "AblationIMV.calculate_imv_matrix" in source


def test_every_notebook_uses_this_repository_package_directly():
    """Reject PyPI fallbacks, path injection, and notebook-local IMV copies."""
    banned_local_implementations = {
        "ll", "get_w", "calculate_imv", "vanilla_imv", "imv_from_probs",
        "imv_from_likelihoods", "information_deficit",
    }

    for path in maintained_notebooks():
        source, tree = notebook_python(path)
        expected_class, expected_methods = EXPECTED_PACKAGE_USAGE[path.parent.name]

        imports_package = any(
            isinstance(node, ast.Import)
            and any(alias.name == "imv" for alias in node.names)
            for node in tree.body
        )
        imports_class = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "imv"
            and any(alias.name == expected_class for alias in node.names)
            for node in tree.body
        )
        called_methods = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        local_functions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        assert imports_package and imports_class, (
            f"{path.name} must import imv and {expected_class} from the top-level package"
        )
        assert expected_methods <= called_methods, (
            f"{path.name} is missing package calls {sorted(expected_methods - called_methods)}"
        )
        assert not banned_local_implementations & local_functions
        assert "Path(imv.__file__).resolve().parent" in source
        assert 'REPOSITORY_ROOT / "src" / "imv"' in source
        assert "IMV_SOURCE != EXPECTED_IMV_SOURCE" in source
        assert "def relative_path(" in source
        assert "def _relative_warning_text(" in source
        assert "Path(tempfile.gettempdir())" in source
        assert "warnings.showwarning = _show_relative_warning" in source
        assert source.index("warnings.showwarning = _show_relative_warning") < source.index(
            "%matplotlib inline"
        )
        assert "from {relative_path(IMV_SOURCE)}" in source
        assert "from {IMV_SOURCE}" not in source
        assert "sys.path" not in source
        assert "PYTHONPATH" not in source
        assert "pip install" not in source
        assert "from imv.utils.core" not in source
        if path.parent.name == "ablation_imv":
            assert "def set_seed" not in source
            assert ".backward(" not in source
            assert "optimizer.step(" not in source
            assert "opt.step(" not in source

    imdb_source, _ = notebook_python(
        ROOT / "examples/ablation_imv/ablation_imv_imdb.ipynb"
    )
    assert "AblationIMV.reduce_bert_layers" in imdb_source


def test_every_notebook_figure_uses_the_three_format_package_exporter():
    maintained = maintained_notebooks()
    assert {path.name for path in maintained} == set(EXPECTED_FIGURE_EXPORTS)

    for path in maintained:
        source, tree = notebook_python(path)
        imports_exporter = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "imv.utils"
            and any(alias.name == "save_figure" for alias in node.names)
            for node in tree.body
        )
        export_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_figure"
        ]

        assert imports_exporter
        assert len(export_calls) == EXPECTED_FIGURE_EXPORTS[path.name]
        assert source.count("relative_path(path, start=ARTIFACTS)") == len(export_calls)
        assert ".savefig(" not in source

    plotting_source = (ROOT / "src/imv/utils/plotting.py").read_text()
    assert 'FIGURE_DPI = 800' in plotting_source
    assert 'FIGURE_FORMATS = ("png", "pdf", "svg")' in plotting_source


def test_notebooks_never_render_absolute_paths():
    """Committed outputs must be portable and must not expose machine paths."""
    for path in maintained_notebooks():
        output = "\n".join((notebook_markdown_text(path), notebook_output_text(path)))
        for style, pattern in ABSOLUTE_PATH_PATTERNS.items():
            match = pattern.search(output)
            assert match is None, (
                f"{path.name} contains a rendered {style} absolute path: "
                f"{match.group(0)!r}"
            )


def test_absolute_path_detector_covers_platform_forms_without_matching_urls():
    examples = {
        "POSIX": "saved at " + "/" + "custom-volume/research/project/result.csv",
        "Windows drive": "saved at " + "C:" + "\\Users\\researcher\\result.csv",
        "Windows UNC": "saved at " + "\\" + "\\server\\share\\result.csv",
        "file URI": "saved at " + "file:" + "//" + "/" + "home/researcher/result.csv",
    }
    for style, text in examples.items():
        assert ABSOLUTE_PATH_PATTERNS[style].search(text)

    url = "downloaded from https://example.org/data/result.csv"
    assert all(pattern.search(url) is None for pattern in ABSOLUTE_PATH_PATTERNS.values())


def test_pytest_warning_locations_are_relative():
    rendered = warnings.formatwarning(
        "portable warning",
        UserWarning,
        str(ROOT / "src/imv/utils/core.py"),
        1,
    )
    assert str(ROOT) not in rendered
    assert rendered.startswith("src/imv/utils/core.py:1:")


def test_requirements_installs_the_complete_notebook_runtime():
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == ["-e .[examples-deep-learning]"]
    pyproject = (ROOT / "pyproject.toml").read_text()
    for requirement in (
        '"jupyter>=1"',
        '"ucimlrepo>=0.0.3"',
        '"xgboost>=2"',
        '"lightgbm>=4"',
        '"torch>=2"',
        '"transformers>=4.40"',
        '"datasets>=2.18"',
        '"imv[examples,deep-learning]"',
    ):
        assert requirement in pyproject


def test_every_example_fetches_its_named_dataset_and_ships_none():
    """Each notebook must fetch its exact dataset and keep all data external."""
    maintained = maintained_notebooks()
    assert {path.name for path in maintained} == set(EXPECTED_DATA_FETCHES)

    for path in maintained:
        source, tree = notebook_python(path)

        constants = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name):
                continue
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass

        function, selector, expected = EXPECTED_DATA_FETCHES[path.name]
        actual = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.id if isinstance(node.func, ast.Name) else None
            if isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called != function:
                continue

            if isinstance(selector, int):
                argument = node.args[selector] if len(node.args) > selector else None
            else:
                argument = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == selector),
                    None,
                )
            if isinstance(argument, ast.Name) and argument.id in constants:
                actual.append(constants[argument.id])
            elif argument is not None:
                try:
                    actual.append(ast.literal_eval(argument))
                except (ValueError, TypeError):
                    pass

        assert expected in actual, (
            f"{path.name} must call {function} for {expected!r}; found {actual!r}"
        )

        assert "Path.cwd()" not in source, f"{path.name} writes artifacts into the checkout"
        assert "IMV_ARTIFACT_CACHE" in source

    excluded_trees = {".git", ".venv", ".tox", "build", "dist", "site", "venv"}
    shipped_data = []
    for path in ROOT.glob("**/*"):
        relative = path.relative_to(ROOT)
        if any(part in excluded_trees for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in DATA_FILE_SUFFIXES:
            shipped_data.append(relative)
    assert not shipped_data, f"data files must not be stored in the repository: {shipped_data}"


def test_every_seeded_model_has_ten_complete_runs():
    maintained = maintained_notebooks()
    assert {path.name for path in maintained} == set(EXPECTED_SEEDED_MODELS)

    for path in maintained:
        _, tree = notebook_python(path)
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SEEDS"
                for target in node.targets
            )
        ]
        assert len(assignments) == 1, f"{path.name} must assign SEEDS exactly once"
        seeds = ast.literal_eval(assignments[0].value)
        assert isinstance(seeds, list) and all(isinstance(seed, int) for seed in seeds)
        assert seeds == EXPECTED_SEEDS, (
            f"{path.name} uses {seeds!r}; expected {EXPECTED_SEEDS!r}"
        )

        seed_loops = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "seed"
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "SEEDS"
        ]
        assert seed_loops, f"{path.name} does not execute a for seed in SEEDS loop"
        assert any(
            isinstance(argument, ast.Name) and argument.id == "seed"
            for loop in seed_loops
            for call in ast.walk(loop)
            if isinstance(call, ast.Call)
            for argument in (
                list(call.args) + [keyword.value for keyword in call.keywords]
            )
        ), f"{path.name} does not pass seed into computation"

        expected_models = EXPECTED_SEEDED_MODELS[path.name]
        if path.parent.name == "ablation_imv":
            variant_assignments = [
                node for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "VARIANTS"
                    for target in node.targets
                )
            ]
            assert len(variant_assignments) == 1
            variants = variant_assignments[0].value
            if isinstance(variants, ast.Dict):
                actual_models = {
                    ast.literal_eval(key) for key in variants.keys if key is not None
                }
            else:
                actual_models = set(ast.literal_eval(variants))
            iterates_models = any(
                isinstance(node, ast.For)
                and (
                    isinstance(node.iter, ast.Name) and node.iter.id == "VARIANTS"
                    or isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Attribute)
                    and isinstance(node.iter.func.value, ast.Name)
                    and node.iter.func.value.id == "VARIANTS"
                    and node.iter.func.attr == "items"
                )
                for loop in seed_loops
                for node in ast.walk(loop)
            )
        else:
            factory_functions = [
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "model_factories"
            ]
            assert len(factory_functions) == 1
            returns = [
                node.value for node in ast.walk(factory_functions[0])
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            ]
            assert len(returns) == 1
            actual_models = {
                ast.literal_eval(key) for key in returns[0].keys if key is not None
            }
            iterates_models = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "items"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "model_factories"
                for loop in seed_loops
                for node in ast.walk(loop)
            )

        assert actual_models == expected_models
        assert iterates_models, f"{path.name} does not run every model inside every seed"

        stored_outputs = notebook_output_text(path)
        assert all(model in stored_outputs for model in expected_models), (
            f"{path.name} does not contain executed output for every model"
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
