"""
segmentation_predict.py
-----------------------
Run food segmentation on a single image.
"""

import argparse
import os

import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from segmentation_model import (
    build_segmentation_model,
    get_device
)


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model(

    checkpoint_path,

    num_classes,

    device

):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(

            f"Segmentation checkpoint not found: "

            f"{checkpoint_path}"

        )


    # Build model

    model = build_segmentation_model(

        num_classes=num_classes,

        device=device,

        pretrained=False

    )


    # Load checkpoint

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    # Load trained weights
    # Ignore auxiliary classifier weights

    model.load_state_dict(

        checkpoint["model_state_dict"],

        strict=False

    )


    # Evaluation mode

    model.eval()


    return model

# ============================================================
# IMAGE TRANSFORM
# ============================================================

def get_transform(
    image_size=512
):

    return transforms.Compose([

        transforms.Resize(

            (image_size, image_size)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ])


# ============================================================
# SEGMENT IMAGE
# ============================================================

def segment_image(

    image_path,

    model,

    device,

    image_size=512

):

    if not os.path.exists(image_path):

        raise FileNotFoundError(

            f"Image not found: {image_path}"

        )


    # Load original image

    original_image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # Transform image

    transform = get_transform(

        image_size

    )


    input_tensor = (

        transform(original_image)

        .unsqueeze(0)

        .to(device)

    )


    # Model prediction

    with torch.no_grad():

        output = model(

            input_tensor

        )


        # Output shape:
        # [1, 51, 512, 512]

        prediction = torch.argmax(

            output,

            dim=1

        )


    segmentation = (

        prediction

        .squeeze(0)

        .cpu()

        .numpy()

    )


    return (

        original_image,

        segmentation

    )


# ============================================================
# CREATE FOOD MASK
# ============================================================

def create_food_mask(
    segmentation
):

    # Class 0 = Background

    food_mask = (

        segmentation != 0

    )


    return food_mask.astype(

        np.uint8

    )


# ============================================================
# APPLY MASK TO IMAGE
# ============================================================

def apply_mask(

    image,

    mask

):

    # Resize original image
    # to segmentation size

    image = image.resize(

        (

            mask.shape[1],

            mask.shape[0]

        )

    )


    image_array = np.array(

        image

    )


    # Apply food mask

    masked_image = (

        image_array

        *

        mask[:, :, None]

    )


    return Image.fromarray(

        masked_image.astype(

            np.uint8

        )

    )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(

    mask,

    output_path

):

    mask_image = Image.fromarray(

        (

            mask * 255

        ).astype(

            np.uint8

        )

    )


    mask_image.save(

        output_path

    )


# ============================================================
# MAIN
# ============================================================

def main():


    parser = argparse.ArgumentParser(

        description=

        "Indian Food Image Segmentation"

    )


    # --------------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------------

    parser.add_argument(

        "--image",

        type=str,

        required=True

    )


    # --------------------------------------------------------
    # SEGMENTATION CHECKPOINT
    # --------------------------------------------------------

    parser.add_argument(

        "--checkpoint",

        type=str,

        default=

        "checkpoints/best_segmentation_model.pt"

    )


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(

        "--output_dir",

        type=str,

        default=

        "segmentation_output"

    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(

        "\nLoading segmentation model..."

    )


    model = load_segmentation_model(

        checkpoint_path=

        args.checkpoint,

        num_classes=51,

        device=device

    )


    print(

        "Segmentation model loaded successfully!"

    )


    # ========================================================
    # SEGMENT IMAGE
    # ========================================================

    print(

        "\nSegmenting image..."

    )


    original_image, segmentation = (

        segment_image(

            image_path=args.image,

            model=model,

            device=device

        )

    )


    # ========================================================
    # CREATE FOOD MASK
    # ========================================================

    food_mask = create_food_mask(

        segmentation

    )


    # ========================================================
    # CREATE SEGMENTED IMAGE
    # ========================================================

    segmented_image = apply_mask(

        original_image,

        food_mask

    )


    # ========================================================
    # CREATE OUTPUT FOLDER
    # ========================================================

    os.makedirs(

        args.output_dir,

        exist_ok=True

    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    mask_path = os.path.join(

        args.output_dir,

        "food_mask.png"

    )


    segmented_path = os.path.join(

        args.output_dir,

        "segmented_food.png"

    )


    # ========================================================
    # SAVE MASK
    # ========================================================

    save_mask(

        food_mask,

        mask_path

    )


    # ========================================================
    # SAVE SEGMENTED IMAGE
    # ========================================================

    segmented_image.save(

        segmented_path

    )


    # ========================================================
    # DETECT CLASSES
    # ========================================================

    unique_classes = np.unique(

        segmentation

    )


    food_classes = unique_classes[

        unique_classes != 0

    ]


    # ========================================================
    # RESULTS
    # ========================================================

    print(

        "\n=============================="

    )

    print(

        "SEGMENTATION COMPLETE"

    )

    print(

        "==============================\n"

    )


    print(

        f"Segmentation classes found: "

        f"{unique_classes}"

    )


    print(

        f"Food classes found: "

        f"{food_classes}"

    )


    print(

        f"\nFood mask saved to:"

    )

    print(

        mask_path

    )


    print(

        f"\nSegmented food image saved to:"

    )

    print(

        segmented_path

    )


if __name__ == "__main__":

    main()