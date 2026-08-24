# Human Activity Recognition Ablation

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_har.ipynb)
adds architecture diversity with a recurrent sequence model on smartphone
inertial signals.

## Download and provenance

The notebook downloads the official
[UCI Human Activity Recognition archive](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
into `IMV_DATA_CACHE/uci_har`, defaulting beneath `~/.cache/imv/datasets`.
Extraction validates member paths and handles UCI's nested original archive. No
signal file is stored in the repository.

## Preprocess

The supplied 128-step windows contain nine body-acceleration, body-gyroscope, and
total-acceleration channels. The binary contrast keeps
`WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS`, with upstairs as one. This yields
2,059 training and 891 test windows.

UCI's official participant split is preserved: 21 training participants and 9
test participants with zero overlap. Each channel is standardized using training
means and standard deviations only, then the same transformation is applied to
test data. This avoids both participant and preprocessing leakage.

## Ablations and seeds

The reference `FullBiGRU` has two bidirectional GRU layers and learned temporal
attention. The four ablations are `UniGRU`, `OneLayer`, `NoAttention`, and
`MeanPoolMLP`; together they remove bidirectionality, recurrent depth, learned
attention, or temporal recurrence entirely.

Every architecture runs under seeds `[42, 43, 44, 45, 46]` via
`AblationIMV.train_and_evaluate`. Training uses Adam, 15 epochs, learning rate
`1e-3`, hidden size 64, batch size 128, and recurrent/head dropout where
applicable. The 25 fits expose initialization, minibatch, and dropout variability.

Restart predictions remain under the external artifact cache and are reused only
after alignment and probability validation.

## IMV outputs

The installed package computes one directional ablation matrix per seed and the
elementwise five-seed mean. The notebook retains each cell by enhanced model,
baseline model, and seed, plus likelihood, information deficit, accuracy,
parameter count, and runtime diagnostics.

Figures use the same semantics as the IMDb and MNIST examples: a full
directional heatmap and a fixed-`FullBiGRU` comparison against each ablated
baseline. Error bars are seed standard deviations, not confidence intervals.
PNG at 800 DPI, PDF, and SVG are exported outside the repository.

