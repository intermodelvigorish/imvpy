"""Keep the published documentation complete and connected to the live API."""

import inspect
from pathlib import Path

import yaml

import imv
import imv.ablation
import imv.binary
import imv.core
import imv.multiclass
import imv.utils

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "documentation"


def _nav_pages(value):
    """Yield Markdown paths from arbitrarily nested MkDocs navigation."""
    if isinstance(value, str):
        if value.endswith(".md"):
            yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _nav_pages(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _nav_pages(item)


def _documentation_text():
    return "\n".join(path.read_text() for path in DOCS.rglob("*.md"))


def test_every_documentation_page_is_in_navigation_and_exists():
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    assert config["docs_dir"] == "documentation"
    assert config["site_dir"] == "site"
    assert config["strict"] is True

    nav_pages = list(_nav_pages(config["nav"]))
    assert len(nav_pages) == len(set(nav_pages)), "MkDocs nav contains duplicate pages"
    assert all((DOCS / page).is_file() for page in nav_pages)

    source_pages = {
        path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")
    }
    assert set(nav_pages) == source_pages


def test_all_exported_symbols_are_documented_and_have_docstrings():
    text = _documentation_text()
    modules = [imv, imv.core, imv.binary, imv.multiclass, imv.ablation, imv.utils]
    for module in modules:
        for name in module.__all__:
            assert name in text, f"{module.__name__}.{name} is absent from documentation"
            value = getattr(module, name)
            if inspect.isfunction(value) or inspect.isclass(value):
                assert inspect.getdoc(value), f"{module.__name__}.{name} has no docstring"


def test_every_public_evaluator_method_is_documented_and_has_a_docstring():
    text = _documentation_text()
    for cls in (imv.BinaryIMV, imv.MulticlassIMV, imv.AblationIMV):
        for name, _descriptor in vars(cls).items():
            if name.startswith("_"):
                continue
            value = getattr(cls, name)
            if not callable(value):
                continue
            assert name in text, f"{cls.__name__}.{name} is absent from documentation"
            assert inspect.getdoc(value), f"{cls.__name__}.{name} has no docstring"


def test_documentation_build_is_a_release_gate():
    pyproject = (ROOT / "pyproject.toml").read_text()
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    readme = (ROOT / "README.md").read_text()

    assert 'docs = [' in pyproject
    assert 'mkdocstrings[python]' in pyproject
    assert "mkdocs build --strict" in workflow
    assert "documentation/index.md" in readme
    assert "mkdocs build --strict" in readme
