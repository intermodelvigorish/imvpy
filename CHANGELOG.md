# Changelog

All notable changes to IMVpy are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 1.2.0 - 2026-09-14

### Added

- A standalone `imvpy` distribution with vanilla, exact SHAP, multiclass, and
  model-ablation workflows.
- Array, pandas `Series`, sequence, and scalar inputs for vanilla IMV.
- Deterministic bracketed inversion of the equivalent-coin likelihood mapping.
- Shared publication figure export to 800-DPI PNG, PDF, and SVG.
- The canonical IMV publication palette, typography, panel geometry, annotated
  bars, framed heatmaps, colorbars, and scoped Matplotlib styling utilities.
- Strict documentation, package, and repository contract tests.
- Trusted Publishing workflows for TestPyPI and PyPI.
- Dedicated CPU-only CI coverage for the optional PyTorch training helpers.

### Changed

- The project is licensed under GNU GPL v3.0 only using the PEP 639
  `GPL-3.0-only` expression.
- The repository is package-only; research replication materials are maintained
  separately.

### Fixed

- Probability validation, below-chance likelihood handling, directional
  ablation output, and multiclass class-column alignment.
