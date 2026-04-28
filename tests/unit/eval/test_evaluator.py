from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lesionshiftai.core.distributed import DistState
from lesionshiftai.eval.evaluator import evaluate_loader, generalization_gap


pytestmark = pytest.mark.unit


class _ToyDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class _ToyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3))


def _rows_with_duplicate_sample() -> list[dict]:
    return [
        {
            "image": torch.ones(3, 4, 4),
            "label": torch.tensor(0.0),
            "sample_id": "S0",
            "dataset": "isic2019",
        },
        {
            "image": torch.ones(3, 4, 4) * 2,
            "label": torch.tensor(1.0),
            "sample_id": "S1",
            "dataset": "isic2019",
        },
        {
            "image": torch.ones(3, 4, 4) * 3,
            "label": torch.tensor(1.0),
            "sample_id": "S1",
            "dataset": "isic2019",
        },
    ]


def test_evaluate_loader_dedupes_predictions(assert_has_metric_keys) -> None:
    loader = DataLoader(_ToyDataset(_rows_with_duplicate_sample()), batch_size=2)
    model = _ToyModel()
    criterion = torch.nn.BCEWithLogitsLoss()

    metrics, preds = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        dist_state=None,
        threshold=0.5,
    )

    assert_has_metric_keys(metrics)
    assert "loss" in metrics
    assert len(preds) == 2
    assert preds["sample_id"].tolist() == ["S0", "S1"]


def test_evaluate_loader_uses_gather_when_distributed(
    monkeypatch: pytest.MonkeyPatch,
    assert_has_metric_keys,
) -> None:
    import lesionshiftai.eval.evaluator as evaluator_mod

    payload_remote = {
        "y_true": [1],
        "y_prob": [0.9],
        "loss_sum": 0.4,
        "n": 1,
        "sample_id": ["R1"],
        "dataset": ["ham10000"],
    }

    def _fake_gather_object(payload):
        return [payload, payload_remote]

    monkeypatch.setattr(evaluator_mod, "all_gather_object", _fake_gather_object)

    rows = [
        {
            "image": torch.ones(3, 4, 4),
            "label": torch.tensor(0.0),
            "sample_id": "L0",
            "dataset": "ham10000",
        }
    ]

    dist_state = DistState(
        enabled=True,
        rank=0,
        world_size=2,
        local_rank=0,
        device=torch.device("cpu"),
    )

    metrics, preds = evaluate_loader(
        model=_ToyModel(),
        loader=DataLoader(_ToyDataset(rows), batch_size=1),
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        dist_state=dist_state,
    )

    assert_has_metric_keys(metrics)
    assert len(preds) == 2
    assert set(preds["sample_id"].tolist()) == {"L0", "R1"}


def test_generalization_gap() -> None:
    val = {"accuracy": 0.8, "f1": 0.7, "roc_auc": 0.9}
    test = {"accuracy": 0.6, "f1": 0.5, "roc_auc": 0.7}
    gap = generalization_gap(val, test)

    assert gap["accuracy_gap_val_minus_test"] == pytest.approx(0.2)
    assert gap["f1_gap_val_minus_test"] == pytest.approx(0.2)
    assert gap["roc_auc_gap_val_minus_test"] == pytest.approx(0.2)
