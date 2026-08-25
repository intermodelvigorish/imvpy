import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from imv import BinaryIMV


def make_data(n=60):
    rng = np.random.RandomState(4)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": (x1 + x2 > 0).astype(int)})


def creator():
    return LogisticRegression(solver="liblinear", random_state=2)


def evaluator(data, **kwargs):
    defaults = dict(data=data, outcome_variable="target",
                    optional_explanatory_variables=["x1", "x2"],
                    model_creator=creator, n_splits=3, n_jobs=1)
    defaults.update(kwargs)
    return BinaryIMV(**defaults)


def test_constructor_does_not_mutate_caller_and_validates_schema():
    data = make_data()
    original = data.copy(deep=True)
    evaluator(data)
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="unknown explanatory"):
        evaluator(data, optional_explanatory_variables=["missing"])
    with pytest.raises(ValueError, match="only binary classification"):
        evaluator(data, model_type="regression")


def test_incomplete_coalitions_warn_and_do_not_rescale_weights():
    from imv import IncompleteCoalitionWarning

    ev = evaluator(make_data())
    full = ev.run_evaluation()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        complete = ev.calculate_imvshapley_value("x1")

    # Truncating the power set previously rescaled every weight, because the
    # player count was inferred from the surviving coalitions.
    ev.all_combinations_imv = {k: v for k, v in full.items() if len(k) <= 1}
    with pytest.warns(IncompleteCoalitionWarning, match="not a valid Shapley value"):
        truncated = ev.calculate_imvshapley_value("x1")
    assert truncated != pytest.approx(complete)

    # Dropping a single coalition must also be reported.
    ev.all_combinations_imv = {k: v for k, v in full.items() if k != ("x2",)}
    with pytest.warns(IncompleteCoalitionWarning, match="1 are missing"):
        ev.calculate_imvshapley_value("x1")


def test_shapley_rejects_unknown_variable_and_empty_results():
    ev = evaluator(make_data())
    ev.run_evaluation()
    with pytest.raises(ValueError, match="not one of the evaluated"):
        ev.calculate_imvshapley_value("nope")
    ev.all_combinations_imv = {}
    with pytest.raises(ValueError, match="run_evaluation"):
        ev.calculate_imvshapley_value("x1")


def test_all_subsets_and_shapley_efficiency_identity():
    ev = evaluator(make_data())
    results = ev.run_evaluation()
    assert set(results) == {(), ("x1",), ("x2",), ("x1", "x2")}
    shapley_sum = sum(ev.calculate_imvshapley_value(v) for v in ["x1", "x2"])
    # Public values are intentionally rounded to three decimals.
    assert shapley_sum == pytest.approx(results[("x1", "x2")][0] - results[()][0], abs=0.002)


def test_stratified_train_test_path_and_invalid_split():
    ev = evaluator(make_data(), split_method="stratified_train_test_split", prop_test=0.25)
    combination, mean, folds = ev.compute_imv_method(("x1",))
    assert combination == ("x1",)
    assert mean == folds[0]
    with pytest.raises(ValueError, match="split_method"):
        evaluator(make_data(), split_method="unknown")
