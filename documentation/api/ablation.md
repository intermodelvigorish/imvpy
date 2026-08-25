# Ablation API

The matrix methods can be called on the class without PyTorch. The constructor,
seeding, training, and DistilBERT surgery require `imv[deep-learning]`.

## Evaluator

::: imv.ablation_imv.AblationIMV
    options:
      members: false

## Reproducibility

::: imv.ablation_imv.AblationIMV.set_seed

## DistilBERT surgery

::: imv.ablation_imv.AblationIMV.reduce_bert_layers

## Training and prediction

::: imv.ablation_imv.AblationIMV.train_and_evaluate

## Directional matrix

::: imv.ablation_imv.AblationIMV.calculate_imv_matrix

## Matrix averaging

::: imv.ablation_imv.AblationIMV.average_imv_matrices
