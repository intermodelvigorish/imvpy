import numpy as np
import pandas as pd
import pytest

from imvpy import AblationIMV


def frame(y, p):
    return pd.DataFrame({"True Label": y, "Positive Probability": p})


def test_analysis_utilities_work_without_torch():
    y = np.array([0, 1, 0, 1])
    predictions = {
        "base": frame(y, np.full(4, 0.5)),
        "good": frame(y, np.array([0.1, 0.9, 0.2, 0.8])),
    }
    matrix = AblationIMV.calculate_imv_matrix(predictions)
    assert matrix.loc["good", "base"] > 0
    assert np.allclose(np.diag(matrix), 0)
    averaged = AblationIMV.average_imv_matrices([matrix, matrix])
    pd.testing.assert_frame_equal(averaged, matrix)
    with pytest.raises(ValueError, match="cannot be empty"):
        AblationIMV.average_imv_matrices([])


def test_legacy_static_core_api():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    assert AblationIMV.ll(y, p) > 0.5
    assert AblationIMV.calculate_imv(p, p, y) == pytest.approx(0)


def test_ablation_analysis_rejects_unaligned_inputs():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="cannot be empty"):
        AblationIMV.calculate_imv_matrix({})
    with pytest.raises(ValueError, match="identical aligned labels"):
        AblationIMV.calculate_imv_matrix({
            "a": frame(y, [0.2, 0.8]),
            "b": frame(y[::-1], [0.8, 0.2]),
        })
    a = pd.DataFrame([[0]], index=["a"], columns=["a"])
    b = pd.DataFrame([[0]], index=["b"], columns=["b"])
    with pytest.raises(ValueError, match="identical index"):
        AblationIMV.average_imv_matrices([a, b])
