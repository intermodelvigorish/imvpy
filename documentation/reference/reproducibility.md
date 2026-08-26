# Reproducibility

## What the package controls

- `BinaryIMV.random_seed` controls shuffled fold or holdout creation.
- `MulticlassIMV.random_state` controls shuffled folds.
- `AblationIMV.set_seed` seeds Python, NumPy, PyTorch CPU, CUDA, and MPS sources
  available through PyTorch.
- The inverse-entropy `brentq` calculation is deterministic for fixed numeric
  inputs.

A seed does not guarantee bit-identical GPU training across hardware, driver,
library, or kernel versions. It makes stochastic inputs controlled enough to
measure remaining variation; it is not proof of determinism.

## Repeated runs

Use enough complete runs to characterize variation when random initialization,
subsampling, fold shuffling, dropout, or estimator randomness can materially
affect conclusions. Ten distinct seeds are a reasonable minimum for many
research analyses, but the appropriate design depends on the estimand and model.

Keep raw seed-level results. Report the mean only with a dispersion summary and
describe that dispersion as run stability. Do not call it a confidence interval
without a valid inferential procedure.

## Environment capture

Record at minimum:

```bash
python --version
python -c "import os, imv; print(imv.__version__, os.path.relpath(imv.__file__))"
python -m pip freeze
```

For deep learning, also record PyTorch, CUDA or MPS, accelerator model, driver,
and relevant model-library versions. Record dataset shapes, class counts, and
device information alongside the analysis outputs.

## Dataset provenance

Pin a provider identifier and version where available, for example OpenML
dataset name plus version or a namespaced Hugging Face dataset ID. Save retrieval
metadata and checksums externally for archival research. A dynamic downloader
guarantees executability, not that a remote provider can never revise an asset.

## Comparison alignment

Within one seed, all ablation variants must score exactly the same held-out rows.
Across seeds, a different seeded sample is acceptable if every within-seed
contrast remains aligned and the sampling procedure is reported. Retain stable
row identifiers outside the two-column scoring frames and assert them before
calculation.

## Documentation and tests

The repository enforces package defaults, public API documentation, and a
data-free standalone boundary through contract tests. A strict MkDocs build
verifies navigation and generated API objects. These checks prevent mechanical
drift but cannot certify the scientific validity of an analysis design.
