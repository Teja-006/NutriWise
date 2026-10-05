"""
segmentation_model.py
---------------------
DeepLabV3-ResNet50 model for Indian food segmentation.
"""

import torch
import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)


# ============================================================
# SEGMENTATION MODEL
# ============================================================

class FoodSegmentationModel(nn.Module):

    def __init__(
        self,
        num_classes=51,
        pretrained=True
    ):

        super().__init__()


        # ----------------------------------------------------
        # LOAD DEEPLABV3 RESNET50
        # ----------------------------------------------------

        weights = (

            DeepLabV3_ResNet50_Weights.DEFAULT

            if pretrained

            else None

        )


        self.model = deeplabv3_resnet50(

            weights=weights,

            weights_backbone=None

        )


        # ----------------------------------------------------
        # REPLACE FINAL CLASSIFIER
        # ----------------------------------------------------

        in_channels = (

            self.model.classifier[4].in_channels

        )


        self.model.classifier[4] = nn.Conv2d(

            in_channels,

            num_classes,

            kernel_size=1

        )


        self.num_classes = num_classes


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        output = self.model(x)

        return output["out"]


# ============================================================
# BUILD MODEL
# ============================================================

def build_segmentation_model(

    num_classes=51,

    device=None,

    pretrained=True

):

    model = FoodSegmentationModel(

        num_classes=num_classes,

        pretrained=pretrained

    )


    if device is not None:

        model = model.to(device)


    return model


# ============================================================
# DEVICE
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

            "CUDA not available."

        )


    return device


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = get_device()


    model = build_segmentation_model(

        num_classes=51,

        device=device,

        pretrained=True

    )


    dummy_input = torch.randn(

        2,

        3,

        512,

        512

    ).to(device)


    with torch.no_grad():

        output = model(

            dummy_input

        )


    print(

        "\n=============================="

    )

    print(

        "SEGMENTATION MODEL TEST"

    )

    print(

        "==============================\n"

    )


    print(

        "Input shape:",

        dummy_input.shape

    )


    print(

        "Output shape:",

        output.shape

    )