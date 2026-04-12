import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class BaselineCNN(nn.Module):
    """Baseline binary classifier using a pretrained ResNet50 backbone."""

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize backbone and replace classifier head with 1-logit output."""
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        # single logit for BCE
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """Return a single logit per sample for BCEWithLogitsLoss."""
        return self.backbone(x).squeeze(1)
