"""Prevent documented YAML defaults from drifting away from Python signatures."""

import inspect
import re
from pathlib import Path

import yaml

from imvpy import AblationIMV, BinaryIMV, MulticlassIMV
from imvpy.utils.core import (
    calculate_imv,
    get_w,
    imv_from_likelihoods,
    ll,
    vanilla_imv,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((ROOT / "config/settings.yaml").read_text())


def defaults(callable_object):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(callable_object).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_version_is_declared_consistently():
    """All user-visible package version declarations must agree."""
    import imvpy

    pyproject = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert imvpy.__version__ == declared
    assert SETTINGS["package_version"] == declared
    assert f"IMVpy {declared} settings reference" in (ROOT / "config/settings.yaml").read_text()
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["version"] == declared


def test_core_settings_match_signatures():
    core = SETTINGS["defaults"]["core_imv"]
    assert defaults(ll)["epsilon"] == core["epsilon"]
    assert defaults(calculate_imv)["epsilon"] == core["epsilon"]
    assert defaults(calculate_imv)["tolerance"] == core["inverse"]["tolerance_gtol"]
    inverse = defaults(get_w)
    assert inverse["guess"] == core["inverse"]["initial_guess"]
    assert [list(pair) for pair in inverse["bounds"]] == core["inverse"]["bounds"]
    assert inverse["tolerance"] == core["inverse"]["tolerance_gtol"]
    assert inverse["method"] == core["inverse"]["method"]
    assert inverse["chance_tolerance_nats"] == core["inverse"]["chance_tolerance_nats"]
    assert defaults(calculate_imv)["method"] == core["inverse"]["method"]
    assert defaults(vanilla_imv)["epsilon"] == core["epsilon"]
    assert defaults(vanilla_imv)["tolerance"] == core["inverse"]["tolerance_gtol"]
    assert defaults(vanilla_imv)["method"] == core["inverse"]["method"]
    assert defaults(imv_from_likelihoods)["tolerance"] == core["inverse"]["tolerance_gtol"]
    assert defaults(imv_from_likelihoods)["method"] == core["inverse"]["method"]


def test_shap_settings_match_constructor():
    documented = SETTINGS["defaults"]["shap_imv"]
    actual = defaults(BinaryIMV)
    for yaml_name, python_name in {
        "split_method": "split_method",
        "n_splits": "n_splits",
        "prop_test": "prop_test",
        "model_type": "model_type",
        "all_combinations_imv": "all_combinations_imv",
        "random_seed": "random_seed",
        "n_jobs": "n_jobs",
        "verbose": "verbose",
    }.items():
        assert documented[yaml_name] == actual[python_name]


def test_multi_settings_match_constructor():
    documented = SETTINGS["defaults"]["multi_imv"]
    actual = defaults(MulticlassIMV)
    for name in [
        "n_splits", "optional_explanatory_variables", "random_state",
        "stratified", "verbose",
    ]:
        assert documented[name] == actual[name]


def test_ablation_settings_match_training_signatures():
    documented = SETTINGS["defaults"]["ablation_imv"]
    assert defaults(AblationIMV)["random_seed"] == documented["random_seed"]
    training = defaults(AblationIMV.train_and_evaluate)
    yaml_training = documented["training"]
    assert training["num_epochs"] == yaml_training["num_epochs"]
    assert training["lr"] == yaml_training["learning_rate"]
    for name in ["optimizer_class", "scheduler_fn", "max_grad_norm", "seed", "verbose"]:
        assert training[name] == yaml_training[name]
    matrix = defaults(AblationIMV.calculate_imv_matrix)
    assert matrix["target_column"] == documented["prediction_columns"]["target"]
    assert matrix["prob_column"] == documented["prediction_columns"]["positive_probability"]
