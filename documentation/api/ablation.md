# Ablation API

The matrix methods can be called on the class without PyTorch. The constructor,
seeding, training, and DistilBERT surgery require `imvpy[deep-learning]`.

## Evaluator

::: imvpy.ablation_imv.AblationIMV
    options:
      members: false

## Reproducibility

::: imvpy.ablation_imv.AblationIMV.set_seed

## DistilBERT surgery

::: imvpy.ablation_imv.AblationIMV.reduce_bert_layers

## Training and prediction

::: imvpy.ablation_imv.AblationIMV.train_and_evaluate

## Directional matrix

::: imvpy.ablation_imv.AblationIMV.calculate_imv_matrix

## Matrix averaging

::: imvpy.ablation_imv.AblationIMV.average_imv_matrices
