"""
IMV (Information Model Vigor) Package

A comprehensive Python package for measuring information gain in machine learning
models using Information Model Vigor (IMV) metrics. Supports binary classification,
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

from .core import ll, get_w, calculate_imv, imv_from_probs
from .binary import BinaryIMV, IMVEvaluator  # IMVEvaluator for backward compatibility
from .multiclass import MulticlassIMV, MultinomialIMV  # MultinomialIMV for backward compatibility

# Optional: AblationIMV requires PyTorch and transformers (not always installed)
try:
    from .ablation import AblationIMV
    _HAS_ABLATION = True
except ImportError:
    _HAS_ABLATION = False
    AblationIMV = None

__version__ = "1.0.0"

__all__ = [
    # Core functions
    "ll",
    "get_w", 
    "calculate_imv",
    "imv_from_probs",
    
    # Main classes
    "BinaryIMV",
    "MulticlassIMV",
    "AblationIMV",
    
    # Backward compatibility aliases
    "IMVEvaluator",
    "MultinomialIMV",
]


def __getattr__(name):
    """
    Lazy import handler for optional dependencies.
    
    Provides helpful error messages when trying to import AblationIMV
    without the required deep learning dependencies.
    """
    if name == "AblationIMV" and not _HAS_ABLATION:
        raise ImportError(
            "AblationIMV requires PyTorch and transformers. "
            "Install with: pip install torch transformers tqdm"
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
