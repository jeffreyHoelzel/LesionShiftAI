"""cnn.py

Basic ViT class setup for baseline.
"""
import torch
import torch.nn as nn
import timm


class ViTBinaryClassifier(nn.Module):
    """
    Implements a binary lesion classifier using a pretrained Vision Transformer backbone.

    Parameters
    ------------
        model_name : str
            Name of the timm Vision Transformer model architecture to create.
        pretrained : bool
            Whether to initialize the backbone with pretrained weights.

    Returns
    --------
        ViTBinaryClassifier : ViTBinaryClassifier
            Binary classification model that outputs one logit per sample.

    Raises
    -------
        RuntimeError
            Raised when the model cannot be created or pretrained weights cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True
    ) -> None:
        """Initializes the Vision Transformer backbone."""
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1
        )

    def forward(self, x: torch.Tensor):
        """
        Runs a forward pass through the Vision Transformer classifier.

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
        return self.backbone(x).squeeze(-1)
