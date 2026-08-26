# Citation and License

## Method

This package is based on:

> Domingue BW, Rahal C, Faul J, Freese J, Kanopka K, Rigos A, et al. (2025).
> “The InterModel Vigorish (IMV) as a flexible and portable approach for
> quantifying predictive accuracy with binary outcomes.” *PLOS ONE*, 20(3),
> e0316491. [https://doi.org/10.1371/journal.pone.0316491](https://doi.org/10.1371/journal.pone.0316491)

When reporting results, cite the method paper and record the package version:

```python
import imv

print(imv.__version__)
```

Also state which extension was used: vanilla IMV, exact SHAP-IMV, multiclass
one-vs-rest/pairwise IMV, or directional ablation IMV. Extensions should not be
attributed to the original paper unless that source explicitly implements them.

## Repository

Use the repository URL for software provenance:

```text
https://github.com/intermodelvigorish/PyIMV
```

For archival publication, cite a tagged release or immutable commit in addition
to the repository landing page.

## License

The package is distributed under the MIT License. See the repository `LICENSE`
file for the full terms.
