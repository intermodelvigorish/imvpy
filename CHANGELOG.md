# Changelog

All notable changes to IMVpy are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- A standalone `imvpy` distribution with vanilla, exact SHAP, multiclass, and
  model-ablation workflows.
- Array, pandas `Series`, sequence, and scalar inputs for vanilla IMV.
- Deterministic bracketed inversion of the equivalent-coin likelihood mapping.
- Shared publication figure export to 800-DPI PNG, PDF, and SVG.
- Strict documentation, package, and repository contract tests.
- Trusted Publishing workflows for TestPyPI and PyPI.
- Dedicated CPU-only CI coverage for the optional PyTorch training helpers.

### Changed

- Package metadata now uses the current PEP 639 license format.
- The repository is package-only; research replication materials are maintained
  separately.

### Fixed

- Probability validation, below-chance likelihood handling, directional
  ablation output, and multiclass class-column alignment.

Before publishing a release, move the relevant entries into a versioned section
with its release date.
