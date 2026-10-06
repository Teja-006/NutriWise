"""
estimate_weight.py
-------------------
Lightweight food weight estimation for CPU/low-memory deployment.

Uses:
1. Plate detection for pixel-to-cm scale.
2. Food segmentation for food area.
3. A lightweight assumed height for volumetric foods.
4. Density lookup from food_weight_lookup.py.

MiDaS depth estimation is intentionally NOT used here because it is
too memory-heavy for low-memory Render instances.

This is a rough estimate for demonstration purposes, not a calibrated
weighing instrument.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from food_classes import CLASSES, NUM_CLASSES as DEFAULT_NUM_CLASSES
from food_weight_lookup import FOOD_PROPERTIES
from predict_food_multiclass_fixed import (
    get_device,
    load_model,
    preprocess,
    detect_items,
)


# ============================================================
# CONFIGURATION
# ============================================================

PLATE_DIAMETER_CM = 27.5

# Default assumed height for volumetric food.
# Tune this if your actual food portions are consistently
# heavier/lighter than expected.
DEFAULT_FOOD_HEIGHT_CM = 1.5


# ============================================================
# PLATE SCALE
# ============================================================

def detect_plate_circle(image_bgr):
    """
    Detect the plate rim using OpenCV Hough circles.

    Returns:
        (cx, cy, radius_px)
        or None if no plate is detected.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_dim // 2, 1),
        param1=100,
        param2=60,
        minRadius=int(min_dim * 0.25),
        maxRadius=int(min_dim * 0.52),
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    # Largest detected circle = assumed plate
    cx, cy, r = max(circles, key=lambda c: c[2])

    return int(cx), int(cy), int(r)


# ============================================================
# MAIN ESTIMATION
# ============================================================

def estimate_weights(
    image_path,
    seg_weights_path,
    plate_diameter_cm=PLATE_DIAMETER_CM,
    height_scale_cm=DEFAULT_FOOD_HEIGHT_CM,
    min_area_percent=1.5,
    min_confidence=0.6,
    device=None,
    **kwargs,
):
    """
    Estimate food weights without MiDaS.

    Parameters:
        image_path:
            Input food image.

        seg_weights_path:
            Path to segmentation model weights.

        plate_diameter_cm:
            Assumed real-world plate diameter.

        height_scale_cm:
            Assumed average food height for volumetric foods.

        min_area_percent:
            Minimum segmentation area.

        min_confidence:
            Minimum confidence.

    Returns:
        results, meta
    """

    device = device or get_device()

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load segmentation model
    # --------------------------------------------------------

    model, num_classes, img_size, normalize = load_model(
        seg_weights_path,
        device,
        default_num_classes=DEFAULT_NUM_CLASSES,
        default_img_size=256,
        default_normalize=False,
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    pil_image = Image.open(image_path).convert("RGB")

    x = preprocess(
        pil_image,
        img_size,
        normalize,
        device,
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(x)["out"][0]

        probs = F.softmax(logits, dim=0).cpu().numpy()

        prediction = np.argmax(probs, axis=0)

    # --------------------------------------------------------
    # Detect food items
    # --------------------------------------------------------

    detected = detect_items(
        probs,
        prediction,
        num_classes,
        min_area_percent,
        min_confidence,
    )

    if not detected:
        return [], None

    # --------------------------------------------------------
    # Resize segmentation to original image size
    # --------------------------------------------------------

    orig_w, orig_h = pil_image.size

    prediction_full = np.array(
        Image.fromarray(
            prediction.astype(np.uint8)
        ).resize(
            (orig_w, orig_h),
            Image.Resampling.NEAREST,
        )
    )

    # --------------------------------------------------------
    # Plate detection
    # --------------------------------------------------------

    image_bgr = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    circle = detect_plate_circle(image_bgr)

    if circle is not None:

        _, _, radius_px = circle

        diameter_px = radius_px * 2

        scale_cm_per_px = (
            plate_diameter_cm / diameter_px
        )

        plate_detected = True

    else:

        # Fallback:
        # assume plate occupies ~90% of the shorter image dimension.

        fallback_diameter_px = (
            min(orig_w, orig_h) * 0.9
        )

        scale_cm_per_px = (
            plate_diameter_cm /
            fallback_diameter_px
        )

        plate_detected = False

    # --------------------------------------------------------
    # Estimate weights
    # --------------------------------------------------------

    results = []

    for class_id, area_pct, confidence in detected:

        name = CLASSES.get(
            class_id,
            f"UNKNOWN_{class_id}",
        )

        mask = prediction_full == class_id

        pixel_count = int(mask.sum())

        area_cm2 = (
            pixel_count *
            (scale_cm_per_px ** 2)
        )

        props = FOOD_PROPERTIES.get(name)

        # ----------------------------------------------------
        # No nutrition/weight information
        # ----------------------------------------------------

        if props is None:

            results.append({
                "class_id": class_id,
                "name": name,
                "area_cm2": round(area_cm2, 1),
                "weight_g": None,
                "note": "no entry in food_weight_lookup.py",
            })

            continue

        # ----------------------------------------------------
        # Flat foods
        # ----------------------------------------------------

        if props["type"] == "flat":

            weight_g = (
                props["typical_weight_g"] *
                (
                    area_cm2 /
                    props["typical_area_cm2"]
                )
            )

        # ----------------------------------------------------
        # Volumetric foods
        # ----------------------------------------------------

        else:

            # Lightweight approximation instead of MiDaS.
            #
            # We assume an average food height.
            # This avoids loading another deep-learning model.

            height_cm = max(
                float(height_scale_cm),
                0.3,
            )

            volume_cm3 = (
                area_cm2 *
                height_cm
            )

            weight_g = (
                volume_cm3 *
                props["density_g_cm3"]
            )

        results.append({
            "class_id": class_id,
            "name": name,
            "area_cm2": round(area_cm2, 1),
            "weight_g": round(weight_g, 1),
            "confidence": round(confidence, 3),
        })

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    meta = {
        "plate_detected": plate_detected,
        "plate_diameter_cm_assumed": plate_diameter_cm,
        "scale_cm_per_px": scale_cm_per_px,
        "depth_estimation": False,
        "assumed_food_height_cm": height_scale_cm,
    }

    return results, meta


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--seg_weights",
        default="checkpoints/food_seg_multiclass_v3.pth",
    )

    parser.add_argument(
        "--plate_diameter_cm",
        type=float,
        default=PLATE_DIAMETER_CM,
    )

    parser.add_argument(
        "--height_scale_cm",
        type=float,
        default=DEFAULT_FOOD_HEIGHT_CM,
    )

    parser.add_argument(
        "--min_area_percent",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
    )

    args = parser.parse_args()

    results, meta = estimate_weights(
        args.image,
        args.seg_weights,
        args.plate_diameter_cm,
        args.height_scale_cm,
        args.min_area_percent,
        args.min_confidence,
    )

    if meta is None:

        print("No items detected.")

        return

    print(
        f"\nPlate scale: "
        f"{'auto-detected' if meta['plate_detected'] else 'FALLBACK (no circle found)'} "
        f"| assumed diameter="
        f"{meta['plate_diameter_cm_assumed']}cm "
        f"| scale="
        f"{meta['scale_cm_per_px']:.4f} cm/px"
    )

    print(
        f"Depth estimation: DISABLED "
        f"| assumed food height="
        f"{meta['assumed_food_height_cm']}cm"
    )

    print("\nEstimated items:")

    for r in results:

        if r["weight_g"] is None:

            print(
                f"  {r['name']:22s} | "
                f"area={r['area_cm2']:6.1f}cm^2 | "
                f"weight=N/A "
                f"({r['note']})"
            )

        else:

            print(
                f"  {r['name']:22s} | "
                f"area={r['area_cm2']:6.1f}cm^2 | "
                f"weight={r['weight_g']:6.1f}g"
            )

    total = sum(
        r["weight_g"]
        for r in results
        if r["weight_g"] is not None
    )

    print(
        f"\nTotal estimated plate weight: "
        f"{total:.1f}g"
    )


if __name__ == "__main__":
    main()