"""Keep the published documentation complete and connected to the live API."""

import inspect
import re
from pathlib import Path

import yaml

import imvpy
import imvpy.ablation
import imvpy.binary
import imvpy.core
import imvpy.multiclass
import imvpy.utils

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
    modules = [
        imvpy,
        imvpy.core,
        imvpy.binary,
        imvpy.multiclass,
        imvpy.ablation,
        imvpy.utils,
    ]
    for module in modules:
        for name in module.__all__:
            assert name in text, f"{module.__name__}.{name} is absent from documentation"
            value = getattr(module, name)
            if inspect.isfunction(value) or inspect.isclass(value):
                assert inspect.getdoc(value), f"{module.__name__}.{name} has no docstring"


def test_every_public_evaluator_method_is_documented_and_has_a_docstring():
    text = _documentation_text()
    for cls in (imvpy.BinaryIMV, imvpy.MulticlassIMV, imvpy.AblationIMV):
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
    contributing = (ROOT / "CONTRIBUTING.md").read_text()

    assert 'docs = [' in pyproject
    assert 'mkdocstrings[python]' in pyproject
    assert "mkdocs build --strict --quiet" in workflow
    assert "https://intermodelvigorish.github.io/imvpy/" in readme
    assert "mkdocs build --strict --quiet" in contributing


def test_documentation_uses_only_relative_local_path_examples():
    pages = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "RELEASING.md",
        ROOT / "SECURITY.md",
        *DOCS.rglob("*.md"),
    ]
    local_absolute_path = re.compile(
        r"(?<![A-Za-z0-9:/.<~])/(?!/)(?:[^/\s<>'\"]+/)+[^/\s<>'\"]*"
    )
    windows_absolute_path = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")

    for page in pages:
        text = page.read_text()
        assert local_absolute_path.search(text) is None, page.relative_to(ROOT)
        assert windows_absolute_path.search(text) is None, page.relative_to(ROOT)
        assert "file:" + "//" not in text.lower(), page.relative_to(ROOT)
        assert "print(imvpy.__file__)" not in text, page.relative_to(ROOT)


def test_pypi_readme_code_and_links_are_portable():
    readme = (ROOT / "README.md").read_text()

    python_blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    assert python_blocks
    for index, source in enumerate(python_blocks, start=1):
        compile(source, f"README.md python block {index}", "exec")

    markdown_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    assert markdown_links
    for target in markdown_links:
        assert target.startswith(("https://", "#")), target
