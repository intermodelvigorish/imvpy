"""
IMV Core: Shared InterModel Vigorish Functions

This module contains the core mathematical functions used across all IMV variants.
All IMV computations (binary, multi-class, ablation) share these base functions.

Mathematical Foundation:
    1. Log-likelihood (ll): Measures prediction quality via geometric mean
    2. Information weight (get_w): Converts likelihood to information weight
    3. IMV calculation: Measures relative information gain

"""

import warnings

import numpy as np
from scipy.optimize import brentq, minimize

# g(w) = w*log(w) + (1-w)*log(1-w) is negative binary entropy, with range
# [-log 2, 0]. So g(w) = log(a) has a real root only for a >= CHANCE_FLOOR.
CHANCE_FLOOR = 0.5

# How far below the floor a likelihood may fall before the boundary value w=0.5
# stops being a usable answer, measured as |log(2a)| nats of residual.
#
# Out-of-fold null models cross the floor routinely, because a constant fitted
# on the training prior is mildly miscalibrated for the test fold. Measured over
# 1500 intercept-only folds, 329 landed below chance with a worst-case deficit of
# 0.27 nats, where w=0.5 is accurate to within that residual. Genuinely broken
# predictors sit orders of magnitude further out: the IMDb NoNorm ablation runs
# 7.2-9.6 nats below chance. This default sits in the gap, above the observed
# sampling noise and far below any real pathology.
DEFAULT_CHANCE_TOLERANCE_NATS = 0.5

# Upper end of the w search interval. g(w) -> 0 as w -> 1, so likelihoods
# arbitrarily close to 1 have genuine roots and the published constraint is
# w <= 1; only w exactly 1 is unusable, where (1-w)*log(1-w) evaluates as
# 0 * -inf. Stopping 1e-12 short keeps the arithmetic finite while raising the
# representable likelihood ceiling from 0.9921 (the historical 0.999 cap) to
# 0.99999999997.
DEFAULT_UPPER_BOUND = 1.0 - 1e-12

# Root-finding backends for the inverse-entropy step; see get_w.
INVERSE_METHODS = ("brentq", "lbfgsb")
DEFAULT_INVERSE_METHOD = "brentq"


def _g(w):
    """Negative binary entropy, the function inverted by get_w."""
    return w * np.log(w) + (1 - w) * np.log(1 - w)


class BelowChanceLikelihoodWarning(UserWarning):
    """A likelihood fell below 0.5, where no equivalent coin weight exists.

    Emitted by :func:`get_w`, which returns NaN in that case. See
    :func:`information_deficit` for a quantity that stays defined there.
    """


def information_deficit(a):
    """
    Excess cross-entropy of a predictor over a fair coin, in nats.

    Defined as ``log(2a)`` for any likelihood ``a > 0``, this stays meaningful
    exactly where the information weight does not. It needs no inverse of the
    entropy function, so it is well defined below the 0.5 chance floor where
    :func:`get_w` returns NaN.

    Args:
        a (float): Geometric mean likelihood from :func:`ll`, in (0, 1]

    Returns:
        float: ``log(2a)`` nats. Positive when the predictor beats a fair coin,
        zero at the chance floor, negative below it.

    Interpretation:
        A value of -7.2 means the predictor's mean log-likelihood is 7.2 nats
        per observation worse than simply calling a fair coin. Unlike IMV, this
        remains finite and ordered for arbitrarily bad predictions, so it is the
        right way to report how bad a below-chance model actually is.

    Examples:
        >>> information_deficit(0.5)      # exactly chance
        0.0
        >>> round(information_deficit(0.00036), 2)   # IMDb NoNorm ablation
        -7.24
    """
    a = float(a)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be a positive finite likelihood")
    return float(np.log(2.0 * a))


def _as_1d_finite(name, values):
    """Return *values* as a non-empty, finite, one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def ll(x, p, epsilon=1e-9):
    """
    Calculate log-likelihood geometric mean for binary predictions.
    
    This is the fundamental measure of prediction quality used in IMV.
    Computes the geometric mean of likelihood values across all samples.
    
    Args:
        x (np.ndarray): True binary labels (0 or 1), shape (n_samples,)
        p (np.ndarray): Predicted probabilities for positive class, shape (n_samples,)
        epsilon (float, optional): Clipping bound that keeps log() finite at
            exact 0/1 predictions. Must lie in (0, 0.5). Default: 1e-9

    Returns:
        float: Geometric mean of likelihood values, in range (0, 1]

    Mathematical Formula:
        p_clipped = clip(p, ε, 1-ε)
        LL(x, p) = exp(mean(x*log(p_clipped) + (1-x)*log(1-p_clipped)))

    Interpretation:
        - Higher values indicate better predictions
        - LL = 1.0: Perfect predictions
        - LL = 0.5: Random guessing
        - LL < 0.5: Worse than random (model is anti-correlated)
        
    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
        >>> ll(y_true, y_pred)
        0.7123...  # Good predictions
        
        >>> y_pred_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        >>> ll(y_true, y_pred_random)
        0.5  # Random guessing
        
    Note:
        - Clips p into [ε, 1-ε] to handle edge cases where p=0 or p=1
        - Clipping (rather than adding ε inside the log) keeps the result
          bounded above by 1, so any epsilon in (0, 0.5) stays valid input
          for get_w()
        - This is a geometric mean, not arithmetic mean
        - Identical implementation used across all IMV modules
    """
    x = _as_1d_finite("x", x)
    p = _as_1d_finite("p", p)
    if x.shape != p.shape:
        raise ValueError("x and p must have the same length")
    if not np.all(np.isin(x, (0.0, 1.0))):
        raise ValueError("x must contain only binary labels 0 and 1")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must contain probabilities in [0, 1]")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be a finite number in (0, 0.5)")
    # Clip into [epsilon, 1-epsilon] rather than adding epsilon inside the log.
    # Adding it lets a perfect predictor score above 1, which is not a valid
    # likelihood and which get_w() then rejects for any epsilon but the default.
    p = np.clip(p, epsilon, 1.0 - epsilon)
    z = np.where(x == 1.0, np.log(p), np.log1p(-p))
    return float(np.exp(np.mean(z)))


def minimize_me(p, a):
    """
    Objective function for information weight optimization.
    
    This function is minimized to find the probability p that corresponds
    to a given likelihood value a. It represents the gap between the
    binary entropy and the target log-likelihood.
    
    Args:
        p (float): Probability value being optimized, in range (0, 1)
        a (float): Target likelihood value from ll() function
        
    Returns:
        float: Absolute difference between entropy and log-likelihood
        
    Mathematical Formula:
        f(p, a) = |p*log(p) + (1-p)*log(1-p) - log(a)|
        
    Note:
        - This is the objective function for get_w()
        - The left term is binary entropy
        - We seek p where entropy equals log-likelihood
        - Used internally by get_w(), not typically called directly
    """
    return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))


def get_w(a, guess=0.5, bounds=[(CHANCE_FLOOR, DEFAULT_UPPER_BOUND)], tolerance=1e-09,  # noqa: B006
          chance_tolerance_nats=DEFAULT_CHANCE_TOLERANCE_NATS,
          method=DEFAULT_INVERSE_METHOD):
    """
    Compute information weight from likelihood value.
    
    Solves the entropy equation for the probability weight w corresponding to a
    given likelihood. This weight represents the information content or
    "certainty" of the model's predictions.

    Args:
        a (float): Likelihood value from ll() function
        guess (float, optional): Starting point for the ``"lbfgsb"`` method only;
            ignored by ``"brentq"``, which needs no starting point. Default: 0.5
        bounds (list of tuples, optional): One ``(lower, upper)`` pair bracketing
            the search. Default: ``[(0.5, 1 - 1e-12)]``
        tolerance (float, optional): Gradient tolerance for the ``"lbfgsb"``
            method only; ignored by ``"brentq"``. Default: 1e-09
        chance_tolerance_nats (float, optional): How far below the 0.5 chance
            floor a likelihood may fall before the weight is reported as
            undefined rather than as the boundary value 0.5. Measured in nats of
            residual, |log(2a)|. Default: 0.5
        method ({"brentq", "lbfgsb"}, optional): Which numerical backend inverts
            the entropy equation. See "Choosing a method" below.
            Default: ``"brentq"``

    Returns:
        float: Information weight w within *bounds*, or NaN when *a* falls
        below the 0.5 chance floor (see Notes)

    Mathematical Background:
        Solves: p*log(p) + (1-p)*log(1-p) = log(a)
        where p is the information weight we seek. The left side is negative
        binary entropy, whose range is [-log 2, 0], so a real solution exists
        only for a in [0.5, 1]. The paper resolves the resulting pair of
        mirrored roots by choosing w >= 1/2.


    Choosing a method:
        g(w) = w*log(w) + (1-w)*log(1-w) is strictly increasing on [0.5, 1), so
        g(w) = log(a) has exactly one solution there. Both backends target that
        same solution but reach it differently.

        ``"brentq"`` (default) is *root finding*. g(lower) - log(a) <= 0 and
        g(upper) - log(a) >= 0, so a sign change brackets the root; Brent's
        method shrinks the bracket until it is machine-narrow. Because the
        bracket provably contains the root at every step, the method cannot
        stall, needs no starting point, and ignores *guess* and *tolerance*.

        ``"lbfgsb"`` is *optimization*: it walks downhill on the non-smooth
        objective |g(w) - log(a)| via minimize_me(). This reproduces results
        published before the backend became selectable. It carries two known
        weaknesses, which is why it is no longer the default:

        1. g'(0.5) = log(0.5/0.5) = 0 exactly, so the lower bound is a
           stationary point of the objective. L-BFGS-B terminates at any
           stationary point, so a step landing on the bound returns 0.5 instead
           of the true root. With the default guess=0.5 this is rare, but
           sweeping guess over [0.5, 0.999] produces a wrong root for roughly
           2% of (a, guess) pairs.
        2. It is about 200x slower, since each iteration needs finite-difference
           gradients of a function whose derivative is available in closed form.

        Both agree to within 5e-9 across the interior of the domain, so
        switching does not move published values except where "lbfgsb" was
        wrong: at a non-default *guess*, or above the historical 0.999 cap.

        Use ``method="lbfgsb"`` together with ``bounds=[(0.5, 0.999)]`` to
        reproduce pre-selectable-backend numbers exactly.

    Interpretation:
        - w = 0.5: No information (random guessing)
        - w = 0.7: Moderate information
        - w = 0.9: High information
        - w -> 1: Near-perfect information


    Examples:
        >>> # Good predictions (high likelihood)
        >>> a_good = 0.8
        >>> w = get_w(a_good)
        >>> print(f"w = {w:.3f}")  # w ≈ 0.92 (high information)
        
        >>> # Poor predictions (low likelihood)
        >>> a_poor = 0.5
        >>> w = get_w(a_poor)
        >>> print(f"w = {w:.3f}")  # w ≈ 0.50 (no information)
        
    Notes:
        - Below a = 0.5 the equation has no solution at all. Within
          chance_tolerance_nats of the floor the boundary value 0.5 is still
          accurate to that residual and is returned, which is the ordinary case
          for out-of-fold null models. Further below, this returns NaN and emits
          BelowChanceLikelihoodWarning rather than asserting the model is
          exactly a fair coin. Use information_deficit() to quantify how far
          below chance the predictions fall.
        - The default upper bound stops 1e-12 short of 1 purely to keep
          (1-w)*log(1-w) finite. The historical 0.999 cap was stricter than
          necessary and silently pinned every likelihood above 0.9921 to the
          same weight; it is not a property of the metric, whose published
          constraint is w <= 1.
        - *guess* and *tolerance* apply to method="lbfgsb" only.
        - For deep-learning-scale precision, prefer method="brentq" (exact to
          machine precision) over tightening *tolerance*.
    """
    a = float(a)
    if not np.isfinite(a) or not 0 < a <= 1:
        raise ValueError("a must be a finite likelihood in (0, 1]")
    if a < CHANCE_FLOOR:
        # g(w) >= -log 2 for every w, so g(w) = log(a) has no root here. Within
        # chance_tolerance_nats the boundary value w=0.5 still reproduces log(a)
        # to that residual, which covers the out-of-fold null models that cross
        # the floor by sampling noise. Beyond it the boundary value is not an
        # approximation of anything, and reporting it would assert the model is
        # exactly a fair coin when in truth no coin is this bad.
        deficit = information_deficit(a)
        if -deficit <= float(chance_tolerance_nats):
            return CHANCE_FLOOR
        warnings.warn(
            f"geometric mean likelihood {a:.6g} is {-deficit:.3f} nats below the "
            f"{CHANCE_FLOOR} chance floor, so no equivalent coin exists and the "
            f"information weight is undefined (returning NaN). Every calibrated "
            f"predictor scores at least {CHANCE_FLOOR}, so this is a certificate of "
            f"miscalibration rather than of weak discrimination. Recalibrate the "
            f"probabilities before comparing models, or use information_deficit() "
            f"to report how far below chance they fall.",
            BelowChanceLikelihoodWarning,
            stacklevel=2,
        )
        return float("nan")
    if len(bounds) != 1 or len(bounds[0]) != 2:
        raise ValueError("bounds must contain one (lower, upper) pair")
    lower, upper = map(float, bounds[0])
    if not (0 < lower <= float(guess) <= upper < 1):
        raise ValueError("bounds and guess must satisfy 0 < lower <= guess <= upper < 1")
    if method not in INVERSE_METHODS:
        raise ValueError(f"method must be one of {list(INVERSE_METHODS)}")

    if method == "brentq":
        target = np.log(a)
        # g is strictly increasing on [0.5, 1), so the bracket holds a root
        # whenever log(a) lies between its endpoints. Above g(upper) the root
        # sits outside the representable interval; report the bound rather than
        # letting brentq raise on a bracket with no sign change.
        if target >= _g(upper):
            return upper
        if target <= _g(lower):
            return lower
        return float(brentq(lambda w: _g(w) - target, lower, upper,
                            xtol=1e-15, rtol=8.9e-16))

    res = minimize(
        minimize_me,
        guess,
        args=(a,),
        options={'ftol': 0, 'gtol': tolerance},
        method='L-BFGS-B',
        bounds=bounds
    )
    if not res.success and not np.isfinite(res.x[0]):
        raise RuntimeError(f"information-weight optimization failed: {res.message}")
    return float(res.x[0])


def calculate_imv(y_basic, y_enhanced, y, epsilon=1e-9, tolerance=1e-09,
                  method=DEFAULT_INVERSE_METHOD):
    """
    Calculate the InterModel Vigorish (IMV) score.
    
    Computes the package's relative transformed-likelihood score when comparing
    an enhanced model against a basic model. It is a bespoke ratio, not a
    percentage of mutual information or likelihood.
    
    Args:
        y_basic (np.ndarray): Predictions from basic/null model, shape (n_samples,)
        y_enhanced (np.ndarray): Predictions from enhanced model with features, shape (n_samples,)
        y (np.ndarray): True binary labels, shape (n_samples,)
        epsilon (float, optional): Probability clipping bound for ll(). Default: 1e-9
        tolerance (float, optional): Optimization tolerance forwarded to get_w();
            used only when method="lbfgsb". Default: 1e-09
        method ({"brentq", "lbfgsb"}, optional): Inverse-entropy backend forwarded
            to get_w(); see that function for the trade-offs. Default: "brentq"


    Returns:
        float: IMV score representing relative information gain
        
    Mathematical Formula:
        IMV = (w_enhanced - w_basic) / w_basic
        
        where:
            w_basic = get_w(ll(y, y_basic))
            w_enhanced = get_w(ll(y, y_enhanced))
            
    Interpretation:
        - IMV > 0: Enhanced model has more information
          * IMV = 0.10 means 10% information gain
          * IMV = 0.50 means 50% information gain
        - IMV = 0: No information gain (models equivalent)
        - IMV < 0: Enhanced model is worse (rare, indicates overfitting)
        
    Examples:
        >>> # Good feature adds information
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_null = np.array([0.6, 0.6, 0.6, 0.6, 0.6])  # Always predicts class frequency
        >>> y_model = np.array([0.9, 0.1, 0.8, 0.85, 0.15])  # Uses features
        >>> imv = calculate_imv(y_null, y_model, y_true)
        >>> print(f"IMV: {imv:.3f}")  # IMV > 0, feature is useful
        
        >>> # Bad feature doesn't help
        >>> y_bad = np.array([0.55, 0.58, 0.62, 0.59, 0.57])  # Noisy predictions
        >>> imv = calculate_imv(y_null, y_bad, y_true)
        >>> print(f"IMV: {imv:.3f}")  # IMV ≈ 0, feature is useless
        
    Use Cases:
        1. **Feature Selection**: Compare models with/without a feature
        2. **Model Comparison**: Compare different architectures
        3. **Ablation Studies**: Measure component importance
        4. **Shapley Values**: Compute marginal contributions
        
    Technical Notes:
        - Works for any model that outputs probabilities
        - Model-agnostic (doesn't depend on model internals)
        - Directional: reversing basic and enhanced changes the denominator
        - Probability-sensitive: calibration and probability scaling matter
        
    Performance:
        - Fast: O(n_samples) complexity
        - Typical runtime: <1ms for 1000 samples
        - Optimization converges in <10 iterations
        
    Note:
        - Ensure predictions are probabilities in [0, 1]
        - For multi-class, use one-vs-rest or pairwise encoding
        - This Bernoulli implementation is not valid for regression
    """
    y_basic = _as_1d_finite("y_basic", y_basic)
    y_enhanced = _as_1d_finite("y_enhanced", y_enhanced)
    y = _as_1d_finite("y", y)
    if not (y_basic.shape == y_enhanced.shape == y.shape):
        raise ValueError("y_basic, y_enhanced, and y must have the same length")
    ll_basic = ll(y, y_basic, epsilon=epsilon)
    ll_enhanced = ll(y, y_enhanced, epsilon=epsilon)
    w0 = get_w(ll_basic, tolerance=tolerance, method=method)
    w1 = get_w(ll_enhanced, tolerance=tolerance, method=method)
    return (w1 - w0) / w0


# Convenience function for backward compatibility
def imv_from_probs(p_basic, p_enhanced, y_true, epsilon=1e-9, tolerance=1e-09):
    """
    Calculate IMV from probability predictions.
    
    Alias for calculate_imv() with more descriptive parameter names.
    Useful when working with probability arrays directly.
    
    Args:
        p_basic (np.ndarray): Predicted probabilities from basic model
        p_enhanced (np.ndarray): Predicted probabilities from enhanced model
        y_true (np.ndarray): True binary labels
        epsilon (float, optional): Smoothing factor. Default: 1e-9
        tolerance (float, optional): Optimization tolerance. Default: 1e-09
        
    Returns:
        float: IMV score
        
    Note:
        This is simply an alias for calculate_imv() with different parameter names.
        Use whichever naming convention is clearer for your use case.
    """
    return calculate_imv(p_basic, p_enhanced, y_true, epsilon=epsilon, tolerance=tolerance)
