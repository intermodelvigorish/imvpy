# Contributing

## Development setup

```bash
git clone https://github.com/intermodelvigorish/PyIMV.git
cd PyIMV
python -m pip install -e ".[dev]"
```

The editable install is required because the package uses a `src/` layout.

## Repository structure

| Path | Purpose |
|---|---|
| `src/imv/utils` | Canonical metric and shared plotting implementations |
| `src/imv/shap_imv` | Exact binary SHAP-IMV evaluator |
| `src/imv/multi_imv` | One-vs-rest and pairwise multiclass evaluator |
| `src/imv/ablation_imv` | PyTorch helper and directional matrix calculation |
| `documentation` | Authored MkDocs sources |
| `config/settings.yaml` | Machine-readable default and profile reference |
| `tests` | Unit, integration, repository, settings, and documentation contracts |

## Validation

Run all merge gates before submitting a change:

```bash
ruff check .
pytest
mkdocs build --strict
python -m build
```

The package test suite is self-contained and does not require network access.

## Public API changes

Keep one canonical implementation for `ll`, `get_w`, and IMV calculation. New
entry points should delegate to it rather than copying numerical logic into an
evaluator. Validate finite values, shape, labels, probability range,
and alignment at public boundaries.

Update all of the following when changing a public signature or behavior:

- Source docstring and generated API page.
- Relevant task guide and API reference.
- `config/settings.yaml` when a default changes.
- Unit tests and settings/documentation contracts.
- Compatibility guidance when existing callers may be affected.

Do not remove a compatibility alias without a documented migration and
deprecation period.

## Documentation changes

Every public root export and public evaluator method must appear in the generated
API reference. Conceptual pages should state scientific limitations separately
from implementation mechanics. Code snippets must use the package's public
imports rather than copied versions of metric functions.

Build in strict mode. Generated HTML under `site/` is ignored and must not be
committed.

## Style

Use ASCII unless a file already requires Unicode. Prefer small, testable changes
and comments that explain a non-obvious decision rather than restating code.
Avoid silent numerical fallback: warnings or explicit `NaN` are preferable when
the metric is mathematically undefined.
