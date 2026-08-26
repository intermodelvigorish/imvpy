import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from imvpy import MulticlassIMV


def creator():
    return LogisticRegression(max_iter=500)


def test_one_vs_all_supports_nonconsecutive_and_string_labels():
    labels = np.array(["ant", "cat", "dog", "ant", "cat", "dog"])
    data = pd.DataFrame({"x": range(6), "target": labels})
    base = np.full((6, 3), 1 / 3)
    enhanced = np.full((6, 3), 0.1)
    for row, label in enumerate(labels):
        enhanced[row, ["ant", "cat", "dog"].index(label)] = 0.8
    enhanced /= enhanced.sum(axis=1, keepdims=True)
    result = MulticlassIMV(data, "target", creator, n_splits=2).one_vs_all_single_fold(
        data, "target", base, enhanced
    )
    assert result["class"].tolist() == ["ant", "cat", "dog"]
    assert (result["imv"] > 0).all()


def test_pairwise_matrix_shape_diagonal_and_validation():
    data = pd.DataFrame({"x": range(6), "target": [10, 20, 30, 10, 20, 30]})
    base = np.full((6, 3), 1 / 3)
    enhanced = base.copy()
    ev = MulticlassIMV(data, "target", creator, n_splits=2)
    matrix = ev.multinominal_imv_matrix(data, "target", base, enhanced)
    assert matrix.index.tolist() == [10, 20, 30]
    assert np.allclose(np.diag(matrix), 0)
    assert np.allclose(matrix, 0, atol=1e-7)
    with pytest.raises(ValueError):
        ev.one_vs_all_single_fold(data, "target", base[:, :2], enhanced[:, :2])


def test_probability_columns_follow_classes_not_fold_membership():
    # Model trained on {0,1,2}; this fold holds only classes 1 and 2, so column
    # position no longer equals fold position. A perfect classifier must still
    # score a large positive IMV rather than reading the wrong columns.
    data = pd.DataFrame({"x": range(8), "target": [1] * 4 + [2] * 4})
    # Uniform base: renormalized over any pair it predicts 0.5, matching the
    # 50/50 split, so the null model sits exactly on the chance floor.
    base = np.tile([1 / 3, 1 / 3, 1 / 3], (8, 1))
    enhanced = np.zeros((8, 3))
    enhanced[:4] = [0.05, 0.90, 0.05]
    enhanced[4:] = [0.05, 0.05, 0.90]
    ev = MulticlassIMV(data, "target", creator, n_splits=2)

    matrix = ev.multinominal_imv_matrix(data, "target", base, enhanced, classes=[0, 1, 2])
    assert matrix.index.tolist() == [0, 1, 2]
    assert matrix.loc[1, 2] > 0.1 and matrix.loc[2, 1] > 0.1
    # Class 0 never appears in this fold, so its pairs are unmeasurable.
    assert np.isnan(matrix.loc[0, 1]) and np.isnan(matrix.loc[1, 0])

    ova = ev.one_vs_all_single_fold(data, "target", base, enhanced, classes=[0, 1, 2])
    assert ova["class"].tolist() == [0, 1, 2]
    assert np.isnan(ova.loc[0, "imv"])
    assert (ova.loc[1:, "imv"] > 0).all()


def test_pairwise_matrix_is_exactly_symmetric():
    """ll() is invariant under (y, p) -> (1-y, 1-p), and pairwise renormalisation
    makes p_j = 1 - p_i, so swapping the pair changes nothing. Unlike the ablation
    matrix, this one carries no directional information."""
    rng = np.random.RandomState(11)
    labels = np.repeat([0, 1, 2, 3], 20)
    data = pd.DataFrame({"x": rng.normal(labels, 0.8), "target": labels})
    probs = rng.dirichlet(np.ones(4), size=len(labels))
    enhanced = rng.dirichlet(np.ones(4) * 2, size=len(labels))
    ev = MulticlassIMV(data, "target", creator, n_splits=2)
    matrix = ev.multinominal_imv_matrix(data, "target", probs, enhanced, classes=[0, 1, 2, 3])
    assert np.nanmax(np.abs(matrix.values - matrix.values.T)) < 1e-12


def test_mismatched_column_count_reports_actionable_error():
    data = pd.DataFrame({"x": range(4), "target": [1, 1, 2, 2]})
    probs = np.tile([0.5, 0.3, 0.2], (4, 1))
    ev = MulticlassIMV(data, "target", creator, n_splits=2)
    with pytest.raises(ValueError, match="classes=model.classes_"):
        ev.one_vs_all_single_fold(data, "target", probs, probs)
    with pytest.raises(ValueError, match="one label per probability column"):
        ev.one_vs_all_single_fold(data, "target", probs, probs, classes=[0, 1])


def test_unstratified_folds_missing_a_class_no_longer_crash():
    # Class 2 is rare enough that plain KFold drops it from several test folds.
    rng = np.random.RandomState(3)
    labels = np.array([0] * 45 + [1] * 45 + [2] * 6)
    x = np.concatenate([rng.normal(0, .3, 45), rng.normal(3, .3, 45), rng.normal(6, .3, 6)])
    data = pd.DataFrame({"x": x, "target": labels})
    ev = MulticlassIMV(data, "target", creator, n_splits=8, random_state=1)

    folds, average = ev.k_fold_one_vs_all()
    assert np.asarray(folds).shape == (8, 3)
    assert average.shape == (3,)
    matrices, mean_matrix = ev.k_fold_imv_matrix()
    assert mean_matrix.shape == (3, 3)
    assert mean_matrix.index.tolist() == [0, 1, 2]


def test_small_stratified_cross_validation_integration():
    rng = np.random.RandomState(8)
    labels = np.repeat(["a", "b", "c"], 15)
    x = np.concatenate([rng.normal(i, 0.3, 15) for i in range(3)])
    data = pd.DataFrame({"x": x, "target": labels})
    ev = MulticlassIMV(data, "target", creator, n_splits=3, random_state=7, stratified=True)
    folds, average = ev.k_fold_one_vs_all()
    assert np.asarray(folds).shape == (3, 3)
    assert average.shape == (3,)
    assert np.all(np.isfinite(average))
