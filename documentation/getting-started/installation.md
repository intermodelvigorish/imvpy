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

Use editable mode while changing the package or running repository notebooks:

```bash
python -m pip install -e ".[dev]"
```

## Optional extras

| Extra | Install command | Purpose |
|---|---|---|
| Base | `python -m pip install .` | Core metric, evaluators, and plotting |
| Progress | `python -m pip install ".[progress]"` | joblib-aware coalition progress |
| Deep learning | `python -m pip install ".[deep-learning]"` | PyTorch training and BERT layer surgery |
| Notebooks | `python -m pip install ".[notebooks]"` | Jupyter execution tools |
| Tabular examples | `python -m pip install ".[examples]"` | Seven tabular notebooks and downloaders |
| All examples | `python -m pip install -r requirements.txt` | All ten notebooks, including deep learning |
| Tests | `python -m pip install ".[test]"` | pytest and contract-test dependencies |
| Documentation | `python -m pip install -e ".[docs]"` | MkDocs site build |
| Development | `python -m pip install -e ".[dev]"` | Tests, docs, lint, build, and notebooks |

PyTorch is lazy-loaded. Calling `vanilla_imv`, computing an ablation matrix from
saved prediction frames, or importing `imv` does not require the deep-learning
extra. Constructing `AblationIMV`, calling its seed/training methods, or reducing
BERT layers does.

## Run the notebooks

Install the complete notebook runtime from the repository root and start Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter lab examples/
```

To execute one notebook non-interactively:

```bash
jupyter nbconvert --execute --to notebook --inplace \
  examples/shap_imv/shap_imv_adult_income.ipynb
```

Every notebook verifies that `imv.__file__` resolves to this checkout's
`src/imv`. This prevents an older globally installed release from silently
producing the results. Package, cache, dataset, warning, and figure locations in
notebook output are always rendered relative to their documented anchor; local
absolute paths are never embedded in the committed notebooks.

## Data and artifact locations

No dataset is stored in the repository. Notebooks download on demand and use
external cache directories:

| Environment variable | Default | Contents |
|---|---|---|
| `IMV_CACHE_HOME` | `~/.cache/imv` | Parent cache directory |
| `IMV_DATA_CACHE` | `$IMV_CACHE_HOME/datasets` | Downloaded public datasets |
| `IMV_ARTIFACT_CACHE` | `$IMV_CACHE_HOME/notebook_artifacts` | CSVs, restart predictions, and figures |

Each figure is exported to the artifact cache as PNG at 800 DPI and as PDF and
SVG. Only the executed notebooks, with their rendered outputs, are committed.

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
