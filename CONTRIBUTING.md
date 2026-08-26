# Contributing

Bug reports, documentation improvements, and focused code contributions are
welcome through [GitHub Issues](https://github.com/intermodelvigorish/imvpy/issues)
and pull requests.

## Development setup

```bash
git clone https://github.com/intermodelvigorish/imvpy.git
cd imvpy
python -m pip install -e ".[dev]"
```

## Required checks

Run the same release gates used in CI:

```bash
ruff check .
python -m pytest --quiet --cov=src/imvpy --cov-report=term-missing
mkdocs build --strict --quiet
check-manifest
python -m build
python -m twine check --strict dist/*
check-wheel-contents dist/*.whl
```

Public behavior changes require tests and documentation. Numerical changes must
state the mathematical or statistical reason and include an independent check
where practical. Do not commit datasets, generated figures, model weights, or
prediction artifacts.

The full contributor guide is available in the
[documentation](https://intermodelvigorish.github.io/imvpy/development/contributing/).
