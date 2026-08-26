# Installation

## Requirements

`imv` supports Python 3.9 through 3.12. The base installation includes NumPy,
pandas, SciPy, scikit-learn, Matplotlib, seaborn, joblib, and tqdm.

From a clone of the repository, install the package itself before importing it.
The project uses a `src/` layout, so adding only the repository root to
`PYTHONPATH` is not a supported installation.

```bash
python -m pip install .
python -c "import os, imv; print(imv.__version__, os.path.relpath(imv.__file__))"
```

Use editable mode while changing the package:

```bash
python -m pip install -e ".[dev]"
```

## Optional extras

| Extra | Install command | Purpose |
|---|---|---|
| Base | `python -m pip install .` | Core metric, evaluators, and plotting |
| Progress | `python -m pip install ".[progress]"` | joblib-aware coalition progress |
| Deep learning | `python -m pip install ".[deep-learning]"` | PyTorch training and BERT layer surgery |
| Tests | `python -m pip install ".[test]"` | pytest and contract-test dependencies |
| Documentation | `python -m pip install -e ".[docs]"` | MkDocs site build |
| Development | `python -m pip install -e ".[dev]"` | Tests, docs, lint, and build tools |

PyTorch is lazy-loaded. Calling `vanilla_imv`, computing an ablation matrix from
saved prediction frames, or importing `imv` does not require the deep-learning
extra. Constructing `AblationIMV`, calling its seed/training methods, or reducing
BERT layers does.

## Build the documentation

```bash
python -m pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

`mkdocs serve` exposes a local development site at `http://127.0.0.1:8000`.
The strict production build writes generated HTML under the ignored `site/`
directory and treats warnings, broken navigation, and unresolved API objects as
failures.

## Conda

The repository also provides a Miniforge-compatible environment:

```bash
conda env create -f environment.yml
conda activate imv
```

`pyproject.toml` remains authoritative for package dependency ranges.
