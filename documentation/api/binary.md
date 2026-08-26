# Binary and SHAP-IMV API

## Evaluator

::: imvpy.shap_imv.BinaryIMV
    options:
      members:
        - calculate_imv_score
        - compute_imv_method
        - run_evaluation
        - plot_single_var_combinations_layered_violin_centralized_zero
        - calculate_weight
        - calculate_imvshapley_value
        - evaluate_imvshapley

## Incomplete coalition warning

::: imvpy.shap_imv.IncompleteCoalitionWarning
    options:
      members: false

`IMVEvaluator` is an identity alias of `BinaryIMV`. It is documented in
[Compatibility](../reference/compatibility.md) but should not be used in new
code.
