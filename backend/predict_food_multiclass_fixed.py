"""
predict_food_multiclass_fixed.py

Prediction script for the NutriWise multi-class food segmentation model.

Checkpoint:
    checkpoints/food_seg_multiclass_v2.pth

The model:
    DeepLabV3 + ResNet50
    42 output classes
    Image size: 256
    Normalization: False

The script:
    1. Loads the trained segmentation model
    2. Runs prediction on an image
    3. Identifies food classes pixel-by-pixel
    4. Filters out tiny/noisy detections
    5. Reports detected foods
    6. Saves a segmentation mask
    7. Saves an overlay image

Usage:

    python predict_food_multiclass_fixed.py

or:

    python predict_food_multiclass_fixed.py \
        --image ITD/test/images/20250616_151617_leftImg8bit.jpg \
        --weights checkpoints/food_seg_multiclass_v2.pth
"""

import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

from food_classes import CLASSES


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# IMPORTANT:
# food_classes.py may contain 51 IDs because your dataset masks
# contain IDs up to 50.
#
# BUT your trained v2 checkpoint was trained with 42 output
# classes.
#
# Therefore this MUST be 42 when loading the current checkpoint.
DEFAULT_NUM_CLASSES = 42

DEFAULT_IMG_SIZE = 256

# The v2 model was trained without ImageNet normalization.
DEFAULT_NORMALIZE = False


# ============================================================
# NORMALIZATION VALUES
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# CLASS NAME HELPER
# ============================================================

def get_class_name(class_id):
    """
    Safely get a class name from CLASSES.

    CLASSES may be:
        - a dictionary
        - a list
        - a tuple

    This prevents errors such as:

        AttributeError:
        'list' object has no attribute 'get'
    """

    # If CLASSES is a dictionary
    if isinstance(CLASSES, dict):
        return CLASSES.get(
            class_id,
            f"UNKNOWN_{class_id}"
        )

    # If CLASSES is a list or tuple
    if isinstance(CLASSES, (list, tuple)):
        if 0 <= class_id < len(CLASSES):
            return CLASSES[class_id]

        return f"UNKNOWN_{class_id}"

    # Fallback
    return f"UNKNOWN_{class_id}"


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    weights_path,
    device,
    default_num_classes,
    default_img_size,
    default_normalize,
    override_num_classes=None,
    override_img_size=None,
    override_normalize=None
):
    """
    Load the trained DeepLabV3 model.

    Supports:

    1. Old checkpoint:
       bare state_dict

    2. New checkpoint:
       dictionary containing:
           model_state_dict
           num_classes
           img_size
           normalize
    """

    print("\nLoading model...")
    print("Weights:", weights_path)

    raw = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # NEW CHECKPOINT FORMAT
    # --------------------------------------------------------

    if isinstance(raw, dict) and "model_state_dict" in raw:

        state_dict = raw["model_state_dict"]

        num_classes = raw.get(
            "num_classes",
            default_num_classes
        )

        img_size = raw.get(
            "img_size",
            default_img_size
        )

        normalize = raw.get(
            "normalize",
            default_normalize
        )

        print(
            f"Loaded new-format checkpoint:"
            f" num_classes={num_classes},"
            f" img_size={img_size},"
            f" normalize={normalize}"
        )

    # --------------------------------------------------------
    # OLD CHECKPOINT FORMAT
    # --------------------------------------------------------

    else:

        state_dict = raw

        num_classes = default_num_classes
        img_size = default_img_size
        normalize = default_normalize

        print(
            "Loaded old-format checkpoint "
            "(no metadata)."
        )

        print(
            f"Assuming:"
            f" num_classes={num_classes},"
            f" img_size={img_size},"
            f" normalize={normalize}"
        )

    # --------------------------------------------------------
    # USER OVERRIDES
    # --------------------------------------------------------

    if override_num_classes is not None:
        num_classes = override_num_classes

    if override_img_size is not None:
        img_size = override_img_size

    if override_normalize is not None:
        normalize = override_normalize

    print("\nFinal model configuration:")
    print("  Number of classes:", num_classes)
    print("  Image size:", img_size)
    print("  Normalize:", normalize)

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return (
        model,
        num_classes,
        img_size,
        normalize
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(
    image,
    img_size,
    normalize,
    device
):
    """
    Resize and convert image to PyTorch tensor.
    """

    # Resize image
    resized = image.resize(
        (img_size, img_size),
        Image.Resampling.BILINEAR
    )

    # Convert to numpy
    x = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    # Apply ImageNet normalization only if required
    if normalize:

        x = (
            x - IMAGENET_MEAN
        ) / IMAGENET_STD

    # HWC -> CHW
    x = torch.from_numpy(
        x
    ).permute(
        2,
        0,
        1
    )

    # Add batch dimension
    x = x.unsqueeze(0)

    # Float tensor
    x = x.float()

    # Move to device
    x = x.to(device)

    return x


# ============================================================
# DETECT FOOD ITEMS
# ============================================================

def detect_items(
    probs,
    prediction,
    num_classes,
    min_area_percent,
    min_confidence
):
    """
    Identify food classes that pass both filters.

    Parameters
    ----------
    probs:
        Softmax probabilities.
        Shape:
            (num_classes, H, W)

    prediction:
        Pixel-level predicted class IDs.
        Shape:
            (H, W)

    min_area_percent:
        Minimum percentage of image occupied by a class.

    min_confidence:
        Minimum average softmax confidence.

    Returns
    -------
    List of:

        (
            class_id,
            area_percent,
            average_confidence
        )
    """

    total_pixels = prediction.size

    results = []

    # Find every class present in the prediction
    for class_id in np.unique(prediction):

        class_id = int(class_id)

        # Background
        if class_id == 0:
            continue

        # Pixels belonging to this class
        mask = prediction == class_id

        pixel_count = mask.sum()

        # Percentage of image
        area_percent = (
            pixel_count / total_pixels
        ) * 100

        # Average confidence of pixels
        avg_confidence = probs[class_id][mask].mean()

        # Apply both filters
        if (
            area_percent >= min_area_percent
            and avg_confidence >= min_confidence
        ):

            results.append(
                (
                    class_id,
                    float(area_percent),
                    float(avg_confidence)
                )
            )

    # Largest regions first
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="NutriWise Multi-Class Food Segmentation Prediction"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/food_seg_multiclass_v3.pth",
        help="Path to trained model weights"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override number of classes"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Override image size"
    )

    parser.add_argument(
        "--normalize",
        type=lambda s: s.lower() == "true",
        default=None,
        help="Override normalization: true/false"
    )

    parser.add_argument(
        "--min_area_percent",
        type=float,
        default=1.5,
        help="Minimum area percentage for detection"
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
        help="Minimum average confidence"
    )

    parser.add_argument(
        "--save_overlay",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--no_save_overlay",
        dest="save_overlay",
        action="store_false"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print("\n========================================")
    print("NutriWise Food Segmentation")
    print("========================================")

    print("\nDevice:", device)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    (
        model,
        num_classes,
        img_size,
        normalize
    ) = load_model(
        args.weights,
        device,

        # IMPORTANT:
        # Force old checkpoint to 42 classes
        default_num_classes=DEFAULT_NUM_CLASSES,

        default_img_size=DEFAULT_IMG_SIZE,

        default_normalize=DEFAULT_NORMALIZE,

        override_num_classes=args.num_classes,

        override_img_size=args.img_size,

        override_normalize=args.normalize
    )

    # --------------------------------------------------------
    # GET IMAGE PATH
    # --------------------------------------------------------

    image_path = args.image

    if image_path is None:

        image_path = input(
            "\nEnter image path: "
        ).strip()

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    print("\nOpening image:")
    print(image_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    # Keep original image
    original = image.copy()

    print(
        "Original image size:",
        original.size
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    x = preprocess(
        image,
        img_size,
        normalize,
        device
    )

    print(
        "Model input size:",
        tuple(x.shape)
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    print("\nRunning segmentation...")

    with torch.no_grad():

        output = model(x)

        # DeepLabV3 output
        logits = output["out"][0]

        # Convert logits to probabilities
        probs = F.softmax(
            logits,
            dim=0
        ).cpu().numpy()

        # Select class with highest probability
        prediction = np.argmax(
            probs,
            axis=0
        )

    print("Segmentation completed.")

    # --------------------------------------------------------
    # DETECT FOOD ITEMS
    # --------------------------------------------------------

    detected = detect_items(
        probs,
        prediction,
        num_classes,
        args.min_area_percent,
        args.min_confidence
    )

    # --------------------------------------------------------
    # PRINT DETECTED ITEMS
    # --------------------------------------------------------

    print(
        f"\nDetected items "
        f"(area >= {args.min_area_percent}%, "
        f"confidence >= {args.min_confidence}):"
    )

    if not detected:

        print(
            "  No food items passed both thresholds."
        )

        print(
            "\nYou can test lower thresholds later "
            "if necessary."
        )

    else:

        for (
            class_id,
            area_pct,
            conf
        ) in detected:

            # FIX:
            # Safely retrieve class name
            name = get_class_name(
                class_id
            )

            print(
                f"  {class_id:2d} | "
                f"{name:22s} | "
                f"area={area_pct:5.2f}% | "
                f"confidence={conf:.3f}"
            )

    # --------------------------------------------------------
    # SHOW FILTERED-OUT CLASSES
    # --------------------------------------------------------

    all_present = set(
        int(c)
        for c in np.unique(prediction)
        if int(c) != 0
    )

    kept = set(
        c
        for c, _, _
        in detected
    )

    filtered_out = (
        all_present - kept
    )

    if filtered_out:

        print(
            "\nFiltered out as noise "
            "(present in raw mask but "
            "failed thresholds):"
        )

        for class_id in sorted(
            filtered_out
        ):

            mask = (
                prediction == class_id
            )

            area_pct = (
                mask.sum()
                / prediction.size
            ) * 100

            conf = (
                probs[class_id][mask]
                .mean()
            )

            # FIX:
            # Safely retrieve class name
            name = get_class_name(
                class_id
            )

            print(
                f"  {class_id:2d} | "
                f"{name:22s} | "
                f"area={area_pct:5.2f}% | "
                f"confidence={conf:.3f}"
            )

    # --------------------------------------------------------
    # SAVE SEGMENTATION MASK + OVERLAY
    # --------------------------------------------------------

    if args.save_overlay:

        print(
            "\nCreating segmentation visualization..."
        )

        # Reproducible random colors
        rng = np.random.default_rng(51)

        # Color palette
        palette = np.zeros(
            (num_classes, 3),
            dtype=np.uint8
        )

        # Background remains black
        palette[1:] = rng.integers(
            40,
            255,
            size=(
                num_classes - 1,
                3
            ),
            dtype=np.uint8
        )

        # Convert class IDs into RGB colors
        color_mask = palette[
            prediction
        ]

        # Resize mask back to original image size
        mask_image = Image.fromarray(
            color_mask
        ).resize(
            original.size,
            Image.Resampling.NEAREST
        )

        # Blend with original image
        overlay = Image.blend(
            original,
            mask_image,
            alpha=0.45
        )

        # Save mask
        mask_image.save(
            "multiclass_mask.png"
        )

        # Save overlay
        overlay.save(
            "multiclass_overlay.jpg"
        )

        print(
            "\nSaved:"
        )

        print(
            "  multiclass_mask.png"
        )

        print(
            "  multiclass_overlay.jpg"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "Prediction complete."
    )

    print(
        "========================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()