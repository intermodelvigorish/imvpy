"""
IMV (InterModel Vigorish) Package

A Python package for computing InterModel Vigorish in machine learning
models using InterModel Vigorish (IMV) metrics. Supports binary classification,
multi-class classification, and deep learning ablation studies.

Key Components:
- BinaryIMV: SHAP-based feature importance using IMV
- MulticlassIMV: Pairwise IMV for multi-class classification
- AblationIMV: Ablation studies for deep learning models

Core Functions:
- ll(): Log-likelihood geometric mean
- get_w(): Information weight calculation
- calculate_imv(): IMV score computation

Example Usage:
    >>> from imv import BinaryIMV
    >>> evaluator = BinaryIMV(data, outcome_col='target', features=['f1', 'f2'])
    >>> results = evaluator.run_evaluation()
"""

from .ablation_imv import AblationIMV
from .multi_imv import MulticlassIMV, MultinomialIMV
from .shap_imv import BinaryIMV, IMVEvaluator, IncompleteCoalitionWarning
from .utils.core import (
    BelowChanceLikelihoodWarning,
    calculate_imv,
    get_w,
    imv_from_probs,
    information_deficit,
    ll,
)

__version__ = "1.2.0"

__all__ = [
    # Core functions
    "ll",
    "get_w",
    "calculate_imv",
    "imv_from_probs",
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
