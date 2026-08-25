# IMDb DistilBERT Ablation

The [executed notebook](https://github.com/intermodelvigorish/imv_ml_package/blob/main/examples/ablation_imv/ablation_imv_imdb.ipynb)
compares five transformer variants on binary sentiment.

## Download and provenance

`datasets.load_dataset("stanfordnlp/imdb")` downloads the
[public IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb) into
`IMV_DATA_CACHE/huggingface`, defaulting beneath `~/.cache/imv/datasets`.
`AutoTokenizer` and the DistilBERT checkpoint use the same external cache. No
text, tokenized data, model weight, or prediction file is committed.

## Preprocess

For each seed, the notebook draws class-balanced samples of 5,000 training and
5,000 test reviews from the provider's fixed splits. Text is tokenized with
`distilbert-base-uncased`, padded or truncated to 256 tokens. Test rows are held
constant across all variants within a seed so pairwise IMV labels and
probabilities remain aligned.

The smaller sample and two-epoch schedule are an explicit executable compute
budget, not the original paper's full training setting.

## Ablations and seeds

Five variants run under each seed in `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`:

| Variant | Change from pretrained DistilBERT |
|---|---|
| `Original` | Six-layer reference architecture |
| `3Layers` | Keeps the first three transformer layers through `AblationIMV.reduce_bert_layers` |
| `NoAttention` | Replaces query, key, and value projections with identities |
| `NoFFN` | Replaces each feed-forward block with an identity |
| `NoNorm` | Replaces self-attention and output layer normalization with identities; uses float64 to prevent overflow in the unnormalized residual gradients |

Every fit is delegated to `AblationIMV.train_and_evaluate` with AdamW, gradient
norm clipping at 1.0, batch size 16, two epochs, and learning rate `2e-5`. The
seed controls subsampling,
classifier initialization, and minibatch order. Fifty fits make runtime
hardware-dependent: approximately two hours on the accelerated reference system
and potentially several hours on CPU.

Restart prediction CSVs live only in the external artifact cache. An existing
variant/seed prediction file is reused only after its schema, labels, row count,
finite probability bounds, and probability sums validate; delete that file to
rerun the fit.


## IMV outputs

The notebook calculates a directional matrix for each seed with the package,
then averages valid matrices. Rows are enhanced variants and columns are
baselines. It also records geometric mean likelihood, information deficit,
accuracy, runtime, and cache status for every fit.

Severe surgery, particularly `NoNorm`, can produce likelihood below 0.5. Such a
variant has no equivalent-coin weight: affected IMV cells remain `NaN`, and the
notebook visualizes `information_deficit` rather than replacing them with a
misleading finite score.

The directional heatmap and diagnostic plot are embedded and exported as
800-DPI PNG, PDF, and SVG.
