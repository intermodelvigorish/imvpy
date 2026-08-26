"""IMVpy: InterModel Vigorish for model comparison and attribution.

The package implements vanilla binary IMV, exact SHAP-IMV, multiclass
extensions, and directional model-ablation comparisons.

Key Components:
- BinaryIMV: SHAP-based feature importance using IMV
- MulticlassIMV: Pairwise IMV for multi-class classification
- AblationIMV: Ablation studies for deep learning models

Core Functions:
- ll(): Log-likelihood geometric mean
- get_w(): Information weight calculation
- calculate_imv(): IMV score computation
- vanilla_imv(): Discoverable entry point for the original binary IMV

Example:
    >>> from imvpy import vanilla_imv
    >>> vanilla_imv(0.5, [0.8, 0.2], [1, 0]) > 0
    True
"""

from .ablation_imv import AblationIMV
from .multi_imv import MulticlassIMV, MultinomialIMV
from .shap_imv import BinaryIMV, IMVEvaluator, IncompleteCoalitionWarning
from .utils.core import (
    BelowChanceLikelihoodWarning,
    calculate_imv,
    get_w,
    imv_from_likelihoods,
    imv_from_probs,
    information_deficit,
    ll,
    vanilla_imv,
)

__version__ = "1.2.0"

__all__ = [
    # Core functions
    "ll",
    "get_w",
    "calculate_imv",
    "vanilla_imv",
    "imv_from_probs",
    "imv_from_likelihoods",
    "information_deficit",
    "BelowChanceLikelihoodWarning",
    "IncompleteCoalitionWarning",

    # Main classes
    "BinaryIMV",
    "MulticlassIMV",
    "AblationIMV",

    # Backward compatibility aliases
    "IMVEvaluator",
    "MultinomialIMV",
]
