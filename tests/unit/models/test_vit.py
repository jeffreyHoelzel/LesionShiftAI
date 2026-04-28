import pytest
import torch
import torch.nn as nn

from lesionshiftai.models.vit import ViTBinaryClassifier


pytestmark = pytest.mark.unit


class _DummyBackbone(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones((batch, 1), dtype=torch.float32)


def test_vit_binary_classifier_constructs_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.models.vit as vit_mod

    captured = {}

    def _fake_create_model(model_name, pretrained, num_classes):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["num_classes"] = num_classes
        return _DummyBackbone()

    monkeypatch.setattr(vit_mod.timm, "create_model", _fake_create_model)

    model = ViTBinaryClassifier(model_name="vit_base_patch16_224", pretrained=False)
    out = model(torch.zeros((4, 3, 16, 16), dtype=torch.float32))

    assert out.shape == (4,)
    assert captured == {
        "model_name": "vit_base_patch16_224",
        "pretrained": False,
        "num_classes": 1,
    }
