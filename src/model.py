"""
model.py
--------
ResNet50-based multi-label Indian food classifier.

The model outputs RAW LOGITS.

Sigmoid is NOT applied inside the model because
BCEWithLogitsLoss applies it internally during training
in a numerically stable way.
"""

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet50_Weights


# ============================================================
# MULTI-LABEL RESNET50
# ============================================================

class MultiLabelResNet50(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3
    ):

        super().__init__()


        # ----------------------------------------------------
        # PRETRAINED WEIGHTS
        # ----------------------------------------------------

        weights = (

            ResNet50_Weights.IMAGENET1K_V2

            if pretrained

            else None

        )


        # ----------------------------------------------------
        # LOAD RESNET50
        # ----------------------------------------------------

        self.backbone = models.resnet50(

            weights=weights

        )


        # ----------------------------------------------------
        # REPLACE FINAL LAYER
        # ----------------------------------------------------

        in_features = self.backbone.fc.in_features


        self.backbone.fc = nn.Sequential(

            nn.Dropout(

                p=dropout

            ),

            nn.Linear(

                in_features,

                num_classes

            )

        )


        self.num_classes = num_classes


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # Returns RAW LOGITS
        #
        # Shape:
        #
        # (batch_size, num_classes)

        return self.backbone(x)


    # ========================================================
    # FREEZE BACKBONE
    # ========================================================

    def freeze_backbone(self):

        """
        Freeze all ResNet layers.

        Only the final classification
        head (fc) remains trainable.
        """

        for name, param in self.backbone.named_parameters():

            param.requires_grad = (

                "fc" in name

            )


    # ========================================================
    # UNFREEZE LAST BLOCKS
    # ========================================================

    def unfreeze_last_blocks(

        self,

        blocks=("layer4",)

    ):

        """
        Unfreeze selected ResNet blocks.

        Example:

        blocks=("layer4",)

        or:

        blocks=("layer3", "layer4")
        """

        for name, param in self.backbone.named_parameters():

            if (

                "fc" in name

                or

                any(

                    block in name

                    for block in blocks

                )

            ):

                param.requires_grad = True


    # ========================================================
    # UNFREEZE ALL
    # ========================================================

    def unfreeze_all(self):

        """
        Unfreeze the complete model.
        """

        for param in self.backbone.parameters():

            param.requires_grad = True


    # ========================================================
    # COUNT TRAINABLE PARAMETERS
    # ========================================================

    def trainable_parameter_count(self):

        return sum(

            p.numel()

            for p in self.backbone.parameters()

            if p.requires_grad

        )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(

    num_classes: int,

    device: torch.device,

    pretrained: bool = True

):


    model = MultiLabelResNet50(

        num_classes=num_classes,

        pretrained=pretrained

    )


    model = model.to(device)


    return model


# ============================================================
# GET DEVICE
# ============================================================

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

            "using CPU. Training will be slow."

        )


    return device