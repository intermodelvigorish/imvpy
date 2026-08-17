# Reproducible examples

All maintained examples are package-version Jupyter notebooks. They import
`imv`; they do not copy the metric implementation into notebook cells.

## Layout

- `shap_imv/`: binary exact SHAP-IMV notebooks, cached result objects, and figures
- `multi_imv/`: multiclass notebooks, cached results, and figures
- `ablation_imv/`: IMDb/ResNet saved predictions, executed analysis notebooks,
  directional and legacy-compatible tables, and figures
- `data/`: processed local datasets used by the notebooks
- `original_notebooks/`: immutable research notebooks used as parity references

Activate the environment and open Jupyter from the repository root:

```bash
conda activate imv
jupyter lab
```

The maintained notebooks resolve paths from their own directory. Automated
execution should therefore set the working directory to the notebook's parent.
Figures are shown inline and saved in each example family's `figures/` folder.

Every maintained notebook executes the installed package and generates its plots
from the values computed in that notebook. SHAP notebooks use a clearly labelled
interactive configuration (five features for most datasets and three folds) so
exact coalition evaluation finishes in practical time. Multiclass notebooks use
three fresh folds. The old pickle files remain only as paper-reference artifacts;
the notebooks do not load them.

The audit additionally includes a fresh full seven-feature, ten-fold Titanic
seed-42 run and a fresh ten-fold Nursery seed-42 run in their respective
`results/*_parity.json` files.
