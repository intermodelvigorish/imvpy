# utils.py
# Common utilities shared by IMV evaluators.

from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize


def bernoulli_geomean_ll(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Geometric-mean Bernoulli likelihood:
        exp( mean( y*log p + (1-y)*log(1-p) ) )
    """
    p = np.clip(p, eps, 1.0 - eps)  #  numerical safety
    z = y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
    return float(np.exp(np.mean(z)))


def _entropy_gap_objective(p: np.ndarray, a: float) -> float:
    """| p log p + (1-p) log(1-p) - log(a) | for scalar p in (0,1)."""
    p = float(p)
    return abs((p * np.log(p)) + ((1.0 - p) * np.log(1.0 - p)) - np.log(a))


def solve_w(a: float, guess: float = 0.75, bounds: Tuple[Tuple[float, float], ...] = ((1e-3, 0.9999),)) -> float:
    """
    Solve w in (0,1) such that:
        p log p + (1-p) log(1-p) ≈ log(a)
    """
    res = minimize(
        _entropy_gap_objective,
        x0=np.array([guess]),
        args=(a,),  # Pass as tuple
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},  #  tighter tolerances
    )
    w = float(res.x[0])
    if not (0.0 < w < 1.0):  #  sanity check
        w = 0.999
    return w


def imv_from_probs(p_basic: np.ndarray, p_enh: np.ndarray, y_true: np.ndarray) -> float:
    """
    IMV = (w1 - w0) / w0, where w is recovered from geometric-mean likelihoods.
    """
    a0 = bernoulli_geomean_ll(y_true, p_basic)
    a1 = bernoulli_geomean_ll(y_true, p_enh)
    w0 = solve_w(a0)
    w1 = solve_w(a1)
    return (w1 - w0) / w0