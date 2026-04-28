import pytest
import torch
import torch.nn as nn

from lesionshiftai.models.cnn import BaselineCNN


pytestmark = pytest.mark.unit


class _DummyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones((batch, 1), dtype=torch.float32)


def test_baseline_cnn_uses_expected_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.models.cnn as cnn_mod

    captured = {}

    def _fake_resnet50(*, weights):
        captured["weights"] = weights
        return _DummyResNet()

    monkeypatch.setattr(cnn_mod, "resnet50", _fake_resnet50)
    model = BaselineCNN(pretrained=True)

    out = model(torch.zeros((3, 3, 8, 8), dtype=torch.float32))
    assert out.shape == (3,)
    assert captured["weights"] == cnn_mod.ResNet50_Weights.IMAGENET1K_V2


def test_baseline_cnn_no_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.models.cnn as cnn_mod

    captured = {}

    def _fake_resnet50(*, weights):
        captured["weights"] = weights
        return _DummyResNet()

    monkeypatch.setattr(cnn_mod, "resnet50", _fake_resnet50)
    _ = BaselineCNN(pretrained=False)
    assert captured["weights"] is None
