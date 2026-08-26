import warnings

import numpy as np
import pandas as pd
import pytest

from imvpy.core import (
    BelowChanceLikelihoodWarning,
    calculate_imv,
    get_w,
    imv_from_likelihoods,
    imv_from_probs,
    information_deficit,
    ll,
    vanilla_imv,
)
from imvpy.utils.core import DEFAULT_UPPER_BOUND


def test_ll_matches_clipped_likelihood_formula():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.9, 0.3])
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    expected = np.exp(np.mean(y * np.log(clipped) + (1-y) * np.log(1-clipped)))
    assert ll(y, p) == pytest.approx(expected, rel=1e-14)
    # The legacy implementation added epsilon inside the log instead of clipping; interior
    # probabilities are untouched by clipping so the two agree to order epsilon.
    legacy = np.exp(np.mean(y * np.log(p + 1e-9) + (1-y) * np.log(1-p + 1e-9)))
    assert ll(y, p) == pytest.approx(legacy, rel=1e-8)


def test_ll_never_exceeds_one_and_accepts_any_valid_epsilon():
    y = np.array([1.0, 0.0, 1.0])
    perfect = np.array([1.0, 0.0, 1.0])
    for epsilon in [1e-9, 1e-8, 1e-6, 1e-4, 0.01]:
        value = ll(y, perfect, epsilon=epsilon)
        assert 0 < value <= 1
        # get_w must accept whatever ll produces, for every valid epsilon.
        assert 0.5 <= get_w(value) <= DEFAULT_UPPER_BOUND
    for bad in [0.0, -1e-9, 0.5, 1.0, np.nan]:
        with pytest.raises(ValueError, match="epsilon"):
            ll(y, perfect, epsilon=bad)


def test_calculate_imv_runs_for_non_default_epsilon():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    basic = np.full(5, 0.6)
    enhanced = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    assert all(
        np.isfinite(calculate_imv(basic, enhanced, y, epsilon=e))
        for e in [1e-9, 1e-6, 1e-4, 0.01, 0.2]
    )
    # Heavier clipping pulls the enhanced model back toward the baseline. Only
    # visible once 1-epsilon drops below the a=0.9921 inverse-map ceiling, which
    # otherwise pins every near-perfect likelihood to the same weight.
    values = [calculate_imv(basic, enhanced, y, epsilon=e) for e in [0.01, 0.05, 0.2]]
    assert values[0] > values[1] > values[2] > 0


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_ll_exact_constant_probability(p):
    y = np.array([0, 1])
    assert np.isfinite(ll(y, np.array([p, p])))


@pytest.mark.parametrize(
    "y,p,message",
    [
        ([], [], "empty"),
        ([0, 1], [0.2], "same length"),
        ([0, 2], [0.2, 0.8], "binary"),
        ([0, 1], [-0.1, 0.8], "probabilities"),
        ([0, 1], [np.nan, 0.8], "finite"),
    ],
)
def test_ll_rejects_invalid_input(y, p, message):
    with pytest.raises(ValueError, match=message):
        ll(y, p)


def test_get_w_reference_points_and_bounds():
    assert get_w(0.5) == pytest.approx(0.5, abs=1e-7)
    assert 0.5 < get_w(0.8) <= DEFAULT_UPPER_BOUND
    # a = 1 is only reachable at w = 1, which is outside the open interval, so
    # the bound itself is the honest answer.
    assert get_w(1.0) == pytest.approx(DEFAULT_UPPER_BOUND)
    with pytest.raises(ValueError):
        get_w(0)


def _exact_root(a):
    """Independent reference: bracketed root of g(w) = log(a) on [0.5, 1)."""
    from scipy.optimize import brentq

    def g(w):
        return w * np.log(w) + (1 - w) * np.log(1 - w)

    return brentq(lambda w: g(w) - np.log(a), 0.5, 1 - 1e-15, xtol=1e-16, rtol=8.9e-16)


@pytest.mark.parametrize("a", [0.51, 0.6, 0.8, 0.95, 0.99, 0.995, 0.999, 0.99999])
def test_brentq_matches_an_independent_root_reference(a):
    assert get_w(a, method="brentq") == pytest.approx(_exact_root(a), abs=1e-12)


def test_raised_upper_bound_no_longer_saturates_confident_models():
    # The retired 0.999 cap pinned every likelihood above ~0.9921 to one value,
    # collapsing genuinely different models onto an identical weight.
    legacy = [get_w(a, bounds=[(0.5, 0.999)], method="lbfgsb") for a in [0.995, 0.999, 0.99999]]
    assert legacy == [pytest.approx(0.999)] * 3
    current = [get_w(a) for a in [0.995, 0.999, 0.99999]]
    assert current[0] < current[1] < current[2]
    assert all(c == pytest.approx(_exact_root(a), abs=1e-12)
               for c, a in zip(current, [0.995, 0.999, 0.99999]))


def test_both_methods_agree_in_the_interior_so_switching_is_safe():
    for a in np.arange(0.51, 0.99, 0.01):
        brent = get_w(float(a), method="brentq")
        lbfgs = get_w(float(a), method="lbfgsb")
        assert brent == pytest.approx(lbfgs, abs=1e-7)


def test_brentq_ignores_guess_where_lbfgsb_can_be_trapped():
    # g'(0.5) = 0 makes the lower bound a stationary point of |g(w) - log a|,
    # which L-BFGS-B can terminate on. brentq has no starting point at all.
    truth = _exact_root(0.55)
    for guess in [0.5, 0.6, 0.9, 0.95, 0.99]:
        assert get_w(0.55, guess=guess, method="brentq") == pytest.approx(truth, abs=1e-12)
    assert get_w(0.55, guess=0.9, method="lbfgsb") == pytest.approx(0.5)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        get_w(0.8, method="newton")


def test_calculate_imv_identity_alias_and_shape_validation():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    assert calculate_imv(p, p, y) == pytest.approx(0.0)
    assert imv_from_probs(p, p, y) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="same length"):
        calculate_imv(p[:-1], p, y)


def test_published_scalar_baseline_case():
    """Reproduce S1-II.1 of Domingue, Rahal, et al. to six decimals."""
    outcomes = np.array([
        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
    enhanced = np.repeat([0.5, 0.9], 20)

    score = vanilla_imv(0.55, enhanced, outcomes)

    assert score == pytest.approx(0.23722913125143966, abs=1e-12)
    assert round(score, 6) == 0.237229
    assert ll(outcomes, 0.55) == pytest.approx(
        ll(outcomes, np.full(outcomes.size, 0.55))
    )


def test_vanilla_imv_accepts_series_lists_and_scalar_observations():
    outcomes = pd.Series([0, 1, 0, 1], name="observed")
    enhanced = pd.Series([0.2, 0.8, 0.3, 0.7], name="enhanced")

    from_series = vanilla_imv(0.5, enhanced, outcomes)
    from_arrays = calculate_imv(
        np.full(4, 0.5),
        enhanced.to_numpy(),
        outcomes.to_numpy(),
    )
    from_lists = imv_from_probs(
        [0.5] * 4,
        enhanced.tolist(),
        outcomes.tolist(),
    )

    assert from_series == pytest.approx(from_arrays)
    assert from_lists == pytest.approx(from_arrays)
    assert vanilla_imv(0.5, 0.9, 1) > 0
    assert ll(1, np.float64(0.9)) == pytest.approx(0.9)


def test_likelihood_entry_points_match_prediction_entry_point():
    outcomes = np.array([0, 1, 0, 1])
    basic = np.full(outcomes.size, 0.5)
    enhanced = np.array([0.2, 0.8, 0.3, 0.7])
    likelihood_basic = ll(outcomes, basic)
    likelihood_enhanced = ll(outcomes, enhanced)
    expected = calculate_imv(basic, enhanced, outcomes)

    assert imv_from_likelihoods(likelihood_basic, likelihood_enhanced) == pytest.approx(
        expected
    )
    assert calculate_imv(likelihood_basic, likelihood_enhanced) == pytest.approx(expected)
    assert vanilla_imv(np.float64(likelihood_basic), likelihood_enhanced) == pytest.approx(
        expected
    )
    assert imv_from_likelihoods(1, 1) == pytest.approx(0.0)


def test_only_true_scalar_predictions_are_broadcast():
    outcomes = np.array([0, 1, 0, 1])
    enhanced = np.array([0.2, 0.8, 0.3, 0.7])

    with pytest.raises(ValueError, match="same length"):
        vanilla_imv(np.array([0.5]), enhanced, outcomes)
    with pytest.raises(ValueError, match="same length"):
        vanilla_imv(pd.Series([0.5]), enhanced, outcomes)
    with pytest.raises(ValueError, match="scalar geometric mean likelihood"):
        imv_from_likelihoods(np.array([0.5]), 0.8)
    with pytest.raises(ValueError, match="scalar geometric mean likelihood"):
        calculate_imv(np.array([0.5]), np.array([0.8]))


def test_vanilla_functions_are_available_from_the_top_level_package():
    import imvpy

    assert imvpy.vanilla_imv is vanilla_imv
    assert imvpy.imv_from_likelihoods is imv_from_likelihoods


def test_below_chance_likelihood_is_nan_not_the_lower_bound():
    # g(w) >= -log 2 for all w, so no root exists below a = 0.5. Returning the
    # bound would claim "exactly a fair coin", which is strictly stronger than
    # the truth: no coin is this bad.
    for a in [0.2, 0.001, 1e-9]:
        with pytest.warns(BelowChanceLikelihoodWarning, match="below the 0.5 chance"):
            assert np.isnan(get_w(a))
    # The floor itself is attainable and must stay a real answer.
    assert get_w(0.5) == pytest.approx(0.5, abs=1e-7)


def test_near_chance_null_models_keep_the_boundary_value():
    # Out-of-fold null models cross the floor by sampling noise; within the
    # tolerance the boundary value still reproduces log(a) to that residual,
    # so IMV must stay a number rather than becoming NaN.
    for a in [0.4999, 0.4977, 0.48, 0.35]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_w(a) == pytest.approx(0.5)
    # The tolerance is a documented convention, not a hard law, so it is tunable.
    with pytest.warns(BelowChanceLikelihoodWarning):
        assert np.isnan(get_w(0.4977, chance_tolerance_nats=0.001))


def test_below_chance_imv_propagates_nan_rather_than_a_fabricated_score():
    y = np.array([1.0, 0.0] * 25)
    baseline = np.full(y.shape, 0.5)
    # Confidently wrong on every observation.
    perverse = np.where(y == 1, 0.001, 0.999)
    with pytest.warns(BelowChanceLikelihoodWarning):
        assert np.isnan(calculate_imv(baseline, perverse, y))
    with pytest.warns(BelowChanceLikelihoodWarning):
        assert np.isnan(calculate_imv(perverse, baseline, y))


def test_information_deficit_stays_defined_and_ordered_below_chance():
    assert information_deficit(0.5) == pytest.approx(0.0)
    assert information_deficit(1.0) == pytest.approx(np.log(2))
    # Strictly decreasing as predictions get worse, unlike the clamped weight.
    worsening = [0.4, 0.1, 0.01, 1e-4, 1e-8]
    deficits = [information_deficit(a) for a in worsening]
    assert all(np.isfinite(deficits))
    assert all(d < 0 for d in deficits)
    assert deficits == sorted(deficits, reverse=True)
    for bad in [0.0, -0.1, np.nan]:
        with pytest.raises(ValueError):
            information_deficit(bad)


def test_better_predictions_have_positive_imv():
    y = np.array([0, 1] * 20)
    baseline = np.full(y.shape, 0.5)
    better = np.where(y == 1, 0.9, 0.1)
    assert calculate_imv(baseline, better, y) > 0
