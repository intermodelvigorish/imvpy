# Reproducibility

## What the package controls

- `BinaryIMV.random_seed` controls shuffled fold or holdout creation.
- `MulticlassIMV.random_state` controls shuffled folds.
- `AblationIMV.set_seed` seeds Python, NumPy, PyTorch CPU, CUDA, and MPS sources
  available through PyTorch.
- Repository model factories also receive each run seed.
- The inverse-entropy `brentq` calculation is deterministic for fixed numeric
  inputs.

A seed does not guarantee bit-identical GPU training across hardware, driver,
library, or kernel versions. It makes stochastic inputs controlled enough to
measure remaining variation; it is not proof of determinism.

## Ten-seed policy

Every repository example runs each stochastic estimator or architecture under
`[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`. Use at least ten complete runs when random
initialization, subsampling, fold shuffling, dropout, or boosted-tree randomness
can materially affect conclusions.

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
transformers, and datasets versions. The example notebooks print the package
source relative to the repository, plus dataset shapes, class counts, and relevant
device information. Any displayed file location is relative to its named anchor.

## Dataset provenance

Pin a provider identifier and version where available, for example OpenML
dataset name plus version or a namespaced Hugging Face dataset ID. Save retrieval
metadata and checksums externally for archival research, while keeping the
repository data-free. A dynamic downloader guarantees executability, not that a
remote provider can never revise an asset.

## Restart caches

Ablation notebooks cache prediction frames outside the repository because a
50-fit run can be expensive. MNIST and HAR validate schema, labels, row count,
finite probability bounds, and probability sums before reuse. IMDb performs the
same checks and uses a training-protocol tag in each seed/variant filename;
delete a file to force retraining after changing model code or dependencies.

Do not treat a stale prediction cache as reproducibility evidence. Include code
version and configuration in cache keys for a production pipeline.

## Comparison alignment

Within one seed, all ablation variants must score exactly the same held-out rows.
Across seeds, a different seeded sample is acceptable if every within-seed
contrast remains aligned and the sampling procedure is reported. Retain stable
row identifiers outside the two-column scoring frames and assert them before
calculation.

## Documentation and tests

The repository enforces defaults, notebook provenance guards, dynamic download
code, ten-seed loops, package imports, output formats, and documentation
coverage through contract tests. A strict MkDocs build verifies navigation and
generated API objects. These checks prevent mechanical drift but cannot certify
the scientific validity of a new dataset or split design.
