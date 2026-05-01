"""cnn.py

Basic CNN class setup for baseline and ensemble.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class BaselineCNN(nn.Module):
    """
    Implements a baseline binary lesion classifier using a ResNet50 backbone.

    Parameters
    ------------
        pretrained : bool
            Whether to initialize the ResNet50 backbone with ImageNet pretrained weights.

    Returns
    --------
        BaselineCNN : BaselineCNN
            Binary classification model that outputs one logit per sample.

    Raises
    -------
        RuntimeError
            Raised when pretrained weights cannot be loaded.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize backbone and replace classifier head with 1-logit output."""
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        # single logit for BCE
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor):
        """
        Runs a forward pass through the baseline CNN.

        Parameters
        ------------
            x : torch.Tensor
                Input image batch tensor.

        Returns
        --------
            logits : torch.Tensor
                Single-logit prediction for each input sample.

        Raises
        -------
            RuntimeError
                Raised when the input tensor shape is incompatible with the backbone.
        """
        return self.backbone(x).squeeze(1)
