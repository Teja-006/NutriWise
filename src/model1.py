"""
model.py
--------
ResNet50-based single-label Indian food classifier.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50FoodClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        self.backbone = models.resnet50(
            weights=weights
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes


    def forward(self, x):

        return self.backbone(x)


    # -----------------------------
    # Transfer Learning Utilities
    # -----------------------------

    def freeze_backbone(self):

        """
        Freeze all ResNet layers except
        the final classification layer.
        """

        for name, param in self.backbone.named_parameters():

            param.requires_grad = "fc" in name


    def unfreeze_last_blocks(
        self,
        blocks=("layer4",)
    ):

        """
        Unfreeze selected ResNet blocks.
        """

        for name, param in self.backbone.named_parameters():

            if (
                "fc" in name
                or any(
                    block in name
                    for block in blocks
                )
            ):
                param.requires_grad = True


    def unfreeze_all(self):

        """Unfreeze the complete model."""

        for param in self.backbone.parameters():

            param.requires_grad = True


    def trainable_parameter_count(self):

        return sum(
            p.numel()
            for p in self.backbone.parameters()
            if p.requires_grad
        )


def build_model(
    num_classes: int,
    device: torch.device,
    pretrained: bool = True
):

    model = ResNet50FoodClassifier(

        num_classes=num_classes,
        pretrained=pretrained

    )

    model = model.to(device)

    return model


def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "CUDA not available, "
            "using CPU."
        )

    return device