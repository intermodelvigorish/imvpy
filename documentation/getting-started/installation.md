# Installation

## Requirements

IMVpy supports Python 3.9 and newer. The base installation includes NumPy,
pandas, SciPy, scikit-learn, Matplotlib, seaborn, joblib, and tqdm.

Install the released distribution from PyPI:

```bash
python -m pip install imvpy
python -c "import imvpy; print(imvpy.__version__)"
```

The project is branded IMVpy, while both the distribution and import package
use the normalized lowercase name `imvpy`.

Use editable mode when working from a clone. The project uses a `src/` layout,
so adding only the repository root to `PYTHONPATH` is not a supported
installation:

```bash
python -m pip install -e ".[dev]"
```

## Optional extras

| Extra | Install command | Purpose |
|---|---|---|
| Base | `python -m pip install imvpy` | Core metric, evaluators, and plotting |
| Progress | `python -m pip install "imvpy[progress]"` | joblib-aware coalition progress |
| Deep learning | `python -m pip install "imvpy[deep-learning]"` | PyTorch training and BERT layer surgery |
| Tests | `python -m pip install ".[test]"` | pytest and contract-test dependencies |
| Documentation | `python -m pip install -e ".[docs]"` | MkDocs site build |
| Release | `python -m pip install -e ".[release]"` | build and distribution validation tools |
| Development | `python -m pip install -e ".[dev]"` | All contributor tools |

PyTorch is lazy-loaded. Calling `vanilla_imv`, computing an ablation matrix from
saved prediction frames, or importing `imvpy` does not require the deep-learning
extra. Constructing `AblationIMV`, calling its seed/training methods, or reducing
BERT layers does.

## Build the documentation

```bash
python -m pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict --quiet
```

`mkdocs serve` exposes a local development site at `http://127.0.0.1:8000`.
The strict production build writes generated HTML under the ignored `site/`
directory and treats warnings, broken navigation, and unresolved API objects as
failures.

## Conda

The repository also provides a Miniforge-compatible environment:

```bash
conda env create -f environment.yml
conda activate imvpy
```

`pyproject.toml` remains authoritative for package dependency ranges.
