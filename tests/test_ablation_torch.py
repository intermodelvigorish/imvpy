import random
from types import SimpleNamespace

import numpy as np
import pytest

from imvpy import AblationIMV

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class TinyBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, features, labels):
        logits = self.classifier(features)
        loss = nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def tiny_loader():
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    records = [
        {"features": features[index], "labels": labels[index]}
        for index in range(len(labels))
    ]
    return DataLoader(records, batch_size=4, shuffle=False)


def test_seed_and_distilbert_layer_reduction(capsys):
    evaluator = AblationIMV(random_seed=17)
    output = capsys.readouterr().out
    assert "Using device:" in output
    assert "PyTorch version:" in output

    evaluator.set_seed()
    first = (random.random(), np.random.random(), torch.rand(1).item())
    evaluator.set_seed()
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == pytest.approx(second)

    model = SimpleNamespace(
        distilbert=SimpleNamespace(
            transformer=SimpleNamespace(layer=nn.ModuleList([nn.Identity() for _ in range(3)]))
        )
    )
    assert AblationIMV.reduce_bert_layers(model, 2) is model
    assert len(model.distilbert.transformer.layer) == 2
    with pytest.raises(ValueError, match="positive integer"):
        AblationIMV.reduce_bert_layers(model, 0)
    with pytest.raises(ValueError, match="model has 2"):
        AblationIMV.reduce_bert_layers(model, 3)


def test_training_helper_returns_aligned_probabilities_and_steps_scheduler():
    evaluator = AblationIMV(random_seed=23)
    train_loader = tiny_loader()
    test_loader = tiny_loader()
    scheduler_steps = []

    class CountingScheduler:
        def step(self):
            scheduler_steps.append(1)

    def scheduler_fn(*, optimizer, num_training_steps):
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert num_training_steps == 4
        return CountingScheduler()

    result = evaluator.train_and_evaluate(
        TinyBinaryClassifier(),
        train_loader,
        test_loader,
        num_epochs=2,
        lr=0.1,
        scheduler_fn=scheduler_fn,
        max_grad_norm=1.0,
        seed=23,
        verbose=False,
    )

    predictions = result["test_predictions"]
    assert list(predictions.columns) == [
        "Negative Probability",
        "Positive Probability",
        "True Label",
        "Predicted Label",
    ]
    assert len(predictions) == 8
    assert np.allclose(
        predictions[["Negative Probability", "Positive Probability"]].sum(axis=1),
        1.0,
    )
    assert len(scheduler_steps) == 4
    for key in ("test_accuracy", "test_precision", "test_recall"):
        assert 0 <= result[key] <= 1

    with pytest.raises(ValueError, match="positive finite scalar"):
        evaluator.train_and_evaluate(
            TinyBinaryClassifier(),
            train_loader,
            test_loader,
            max_grad_norm=0,
            verbose=False,
        )
