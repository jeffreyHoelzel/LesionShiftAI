import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lesionshiftai.core.distributed import DistState
from lesionshiftai.train.engine import train_one_epoch


pytestmark = pytest.mark.unit


class _ToyTrainDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int):
        image = torch.full((3, 8, 8), float(idx + 1) / 8.0)
        label = torch.tensor(float(idx % 2), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "sample_id": f"S{idx}",
            "dataset": "isic2019",
        }


class _TinyTrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3)) * self.scale + self.bias


def test_train_one_epoch_updates_model_and_returns_metrics(assert_has_metric_keys) -> None:
    model = _TinyTrainModel()
    before = model.scale.detach().clone()

    metrics = train_one_epoch(
        model=model,
        loader=DataLoader(_ToyTrainDataset(), batch_size=4),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        dist_state=None,
    )

    assert_has_metric_keys(metrics)
    assert "loss" in metrics
    assert not torch.allclose(before, model.scale.detach())


def test_train_one_epoch_uses_gather_when_distributed(monkeypatch: pytest.MonkeyPatch, assert_has_metric_keys) -> None:
    import lesionshiftai.train.engine as engine_mod

    def _fake_all_gather(payload):
        return [payload, payload]

    monkeypatch.setattr(engine_mod, "all_gather_object", _fake_all_gather)

    dist_state = DistState(
        enabled=True,
        rank=0,
        world_size=2,
        local_rank=0,
        device=torch.device("cpu"),
    )

    model = _TinyTrainModel()
    metrics = train_one_epoch(
        model=model,
        loader=DataLoader(_ToyTrainDataset(), batch_size=4),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        dist_state=dist_state,
    )

    assert_has_metric_keys(metrics)
