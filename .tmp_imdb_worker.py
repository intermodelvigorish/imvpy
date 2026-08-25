import gc
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from imv import AblationIMV

N_TRAIN, N_TEST = 5000, 5000
MAX_LEN, BATCH, EPOCHS, LR, MAX_GRAD_NORM = 256, 16, 2, 2e-5, 1.0
CACHE_TAG = "v2_gradclip1"
NO_NORM_CACHE_TAG = "v3_gradclip1_float64"
COLUMNS = [
    "Negative Probability",
    "Positive Probability",
    "True Label",
    "Predicted Label",
]


class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def valid_predictions(path, expected_labels):
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if list(frame.columns) != COLUMNS or len(frame) != len(expected_labels):
        return False
    probabilities = frame[COLUMNS[:2]].to_numpy(float)
    labels = frame["True Label"].to_numpy(int)
    predicted = frame["Predicted Label"].to_numpy(int)
    return bool(
        np.isfinite(probabilities).all()
        and not np.any((probabilities < 0) | (probabilities > 1))
        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        and np.array_equal(labels, expected_labels)
        and np.array_equal(predicted, probabilities.argmax(axis=1))
    )


def run_seed(seed):
    torch.set_num_threads(int(os.environ.get("IMV_TORCH_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A process-pool worker may execute more than one seed sequentially.
        pass
    cache = Path(os.environ["IMV_CACHE_HOME"])
    data_home = cache / "datasets" / "huggingface"
    results = cache / "notebook_artifacts" / "ablation_imv_imdb" / "results"
    results.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("stanfordnlp/imdb", cache_dir=data_home)
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir=data_home
    )

    def encode(split, n):
        frame = raw[split].to_pandas()
        per_class = n // 2
        parts = [
            frame[frame.label == label].sample(per_class, random_state=seed)
            for label in (0, 1)
        ]
        frame = (
            pd.concat(parts)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        encodings = tokenizer(
            list(frame.text),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        labels = torch.tensor(frame.label.values, dtype=torch.long)
        return IMDbDataset(encodings, labels)

    def original(model):
        return model

    def three_layers(model):
        return AblationIMV.reduce_bert_layers(model, num_layers_to_keep=3)

    def no_attention(model):
        for layer in model.distilbert.transformer.layer:
            layer.attention.q_lin = torch.nn.Identity()
            layer.attention.k_lin = torch.nn.Identity()
            layer.attention.v_lin = torch.nn.Identity()
        return model

    def no_ffn(model):
        for layer in model.distilbert.transformer.layer:
            layer.ffn = torch.nn.Identity()
        return model

    def no_norm(model):
        for layer in model.distilbert.transformer.layer:
            layer.sa_layer_norm = torch.nn.Identity()
            layer.output_layer_norm = torch.nn.Identity()
        return model.double()

    variants = {
        "Original": original,
        "3Layers": three_layers,
        "NoAttention": no_attention,
        "NoFFN": no_ffn,
        "NoNorm": no_norm,
    }
    selected = os.environ.get("IMV_VARIANTS")
    if selected:
        requested = selected.split(",")
        unknown = set(requested) - set(variants)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        variants = {name: variants[name] for name in requested}
    train_ds = encode("train", N_TRAIN)
    test_ds = encode("test", N_TEST)
    expected_labels = test_ds.labels.numpy()
    ablator = AblationIMV(random_seed=seed)

    for name, surgery in variants.items():
        tag = NO_NORM_CACHE_TAG if name == "NoNorm" else CACHE_TAG
        path = results / f"predictions_{tag}_{name}_seed_{seed}.csv"
        if valid_predictions(path, expected_labels):
            print(f"seed {seed} {name} reused", flush=True)
            continue
        ablator.set_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2, cache_dir=data_home
            )
        model = surgery(model)
        run = ablator.train_and_evaluate(
            model,
            DataLoader(train_ds, batch_size=BATCH, shuffle=True),
            DataLoader(test_ds, batch_size=64),
            num_epochs=EPOCHS,
            lr=LR,
            optimizer_class=torch.optim.AdamW,
            max_grad_norm=MAX_GRAD_NORM,
            seed=seed,
            verbose=False,
        )
        run["test_predictions"].to_csv(path, index=False)
        if not valid_predictions(path, expected_labels):
            raise RuntimeError(f"seed {seed} {name} failed prediction validation")
        print(f"seed {seed} {name} complete", flush=True)
        del run, model
        gc.collect()
    return seed


if __name__ == "__main__":
    seeds = [int(value) for value in sys.argv[1:]]
    if not seeds or any(seed not in range(42, 52) for seed in seeds):
        raise SystemExit("pass one or more seeds from 42 through 51")
    context = multiprocessing.get_context("spawn")
    worker_count = min(int(os.environ.get("IMV_WORKERS", "4")), len(seeds))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        for completed_seed in pool.map(run_seed, seeds):
            print(f"seed {completed_seed} finished", flush=True)
