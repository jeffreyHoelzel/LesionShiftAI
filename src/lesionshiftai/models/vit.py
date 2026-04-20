import torch.nn as nn
import timm


class ViTBinaryClassifier(nn.Module):
    """Binary lesion classifier backed by a pretrained ViT model."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1
        )

    def forward(self, x):
        """Return a single logit per sample for BCEWithLogitsLoss."""
        return self.backbone(x).squeeze(-1)
