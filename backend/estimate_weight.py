"""
estimate_weight.py
-------------------
Turns segmented food regions into weight estimates (grams), with NO
reference object required in the photo. Three assumptions make this
work, all documented here so you can tune/justify them:

1. PLATE AS THE SCALE REFERENCE. Standard steel thali plates are a
   fairly consistent ~27-28cm diameter. We auto-detect the plate's
   circular rim in the photo (OpenCV Hough circle transform) and use
   its pixel diameter vs. the assumed real diameter to get a cm/px
   scale factor. If circle detection fails (busy background, cropped
   plate, non-circular tray), we fall back to assuming the plate fills
   ~90% of the image's shorter dimension — worse, but keeps the
   pipeline from crashing. --plate_diameter_cm lets you correct the
   assumption if your plates are a different size.

2. DEPTH AS RELATIVE HEIGHT. MiDaS gives relative (not metric) depth —
   it tells you food region A looks "closer to camera / taller" than
   region B, but not how many cm that actually is. We anchor "zero
   height" to the median depth of background/plate-surface pixels, then
   scale the food regions' depth-above-baseline by --height_scale_cm,
   a single tunable constant (default 3.0, meaning the tallest pile
   in a typical photo is assumed to be ~3cm). This is the single
   biggest source of error in this pipeline — if you can weigh a few
   real plates and compare, adjust this constant to match.

3. DENSITY LOOKUP (food_weight_lookup.py) converts area x height
   (=volume) into weight for most dishes, and uses a per-piece
   weight x area-ratio approach for flat items (roti, papad) where
   depth doesn't resolve a few mm of thickness reliably.

Treat all outputs as rough estimates for a demo, not a calibrated
instrument.

Usage:
    python estimate_weight.py --image path/to/thali.jpg \
        --seg_weights checkpoints/food_seg_multiclass_v2.pth

Requires: opencv-python, and MiDaS (auto-downloaded via torch.hub the
first time you run this — needs a real internet connection, ~100MB).
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from food_classes import CLASSES, NUM_CLASSES as DEFAULT_NUM_CLASSES
from NutriWise.backend.food_weight_lookup import FOOD_PROPERTIES
from NutriWise.backend.predict_food_multiclass_fixed import get_device, load_model, preprocess, detect_items

PLATE_DIAMETER_CM = 27.5
HEIGHT_SCALE_CM = 3.0

_MIDAS_CACHE = {}


# ============================================================
# PLATE SCALE
# ============================================================
def detect_plate_circle(image_bgr):
    """Returns (cx, cy, radius_px) of the plate rim, or None if not found."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dim // 2,
        param1=100, param2=60,
        minRadius=int(min_dim * 0.25), maxRadius=int(min_dim * 0.52),
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])  # largest circle found = plate rim
    return int(cx), int(cy), int(r)


# ============================================================
# DEPTH (MiDaS, lazy-loaded once per process)
# ============================================================
def load_midas(device):
    if "model" in _MIDAS_CACHE:
        return _MIDAS_CACHE["model"], _MIDAS_CACHE["transform"]

    print("Loading MiDaS depth model (first run downloads ~100MB)...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
    model.to(device)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)

    _MIDAS_CACHE["model"] = model
    _MIDAS_CACHE["transform"] = transforms.small_transform
    return model, transforms.small_transform


@torch.no_grad()
def compute_depth_map(midas_model, transform, image_rgb_np, device):
    """Relative inverse-depth map resized to (H, W) of the input image.
    Higher value = closer to camera (taller food pile)."""
    input_batch = transform(image_rgb_np).to(device)
    prediction = midas_model(input_batch)
    prediction = F.interpolate(
        prediction.unsqueeze(1), size=image_rgb_np.shape[:2],
        mode="bicubic", align_corners=False,
    ).squeeze()
    return prediction.cpu().numpy()


# ============================================================
# MAIN ESTIMATION
# ============================================================
def estimate_weights(image_path, seg_weights_path,
                      plate_diameter_cm=PLATE_DIAMETER_CM, height_scale_cm=HEIGHT_SCALE_CM,
                      min_area_percent=1.5, min_confidence=0.6, device=None,
                      midas_loader=load_midas, depth_fn=compute_depth_map):
    """
    midas_loader / depth_fn are swappable for testing without a real
    MiDaS download — see estimate_weight_test.py.
    """
    device = device or get_device()

    model, num_classes, img_size, normalize = load_model(
        seg_weights_path, device,
        default_num_classes=DEFAULT_NUM_CLASSES, default_img_size=256, default_normalize=False,
    )

    pil_image = Image.open(image_path).convert("RGB")
    x = preprocess(pil_image, img_size, normalize, device)

    with torch.no_grad():
        logits = model(x)["out"][0]
        probs = F.softmax(logits, dim=0).cpu().numpy()
        prediction = np.argmax(probs, axis=0)

    detected = detect_items(probs, prediction, num_classes, min_area_percent, min_confidence)
    if not detected:
        return [], None

    orig_w, orig_h = pil_image.size
    prediction_full = np.array(
        Image.fromarray(prediction.astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)
    )

    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    circle = detect_plate_circle(image_bgr)
    if circle is not None:
        _, _, radius_px = circle
        scale_cm_per_px = plate_diameter_cm / (radius_px * 2)
        plate_detected = True
    else:
        fallback_diameter_px = min(orig_w, orig_h) * 0.9
        scale_cm_per_px = plate_diameter_cm / fallback_diameter_px
        plate_detected = False

    midas_model, transform = midas_loader(device)
    depth_map = depth_fn(midas_model, transform, np.array(pil_image), device)

    background_mask = prediction_full == 0
    baseline_depth = np.median(depth_map[background_mask]) if background_mask.sum() > 0 else np.median(depth_map)
    depth_range = max(np.percentile(depth_map, 95) - np.percentile(depth_map, 5), 1e-6)

    results = []
    for class_id, area_pct, confidence in detected:
        name = CLASSES.get(class_id, f"UNKNOWN_{class_id}")
        mask = prediction_full == class_id
        pixel_count = int(mask.sum())
        area_cm2 = pixel_count * (scale_cm_per_px ** 2)

        props = FOOD_PROPERTIES.get(name)
        if props is None:
            results.append({"class_id": class_id, "name": name, "area_cm2": round(area_cm2, 1),
                             "weight_g": None, "note": "no entry in food_weight_lookup.py"})
            continue

        if props["type"] == "flat":
            weight_g = props["typical_weight_g"] * (area_cm2 / props["typical_area_cm2"])
        else:
            region_depth = np.median(depth_map[mask])
            relative_height = max(region_depth - baseline_depth, 0.0)
            height_cm = max((relative_height / depth_range) * height_scale_cm, 0.3)
            weight_g = area_cm2 * height_cm * props["density_g_cm3"]

        results.append({"class_id": class_id, "name": name, "area_cm2": round(area_cm2, 1),
                         "weight_g": round(weight_g, 1), "confidence": round(confidence, 3)})

    meta = {"plate_detected": plate_detected, "plate_diameter_cm_assumed": plate_diameter_cm,
            "scale_cm_per_px": scale_cm_per_px}
    return results, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--seg_weights", default="checkpoints/food_seg_multiclass_v2.pth")
    parser.add_argument("--plate_diameter_cm", type=float, default=PLATE_DIAMETER_CM)
    parser.add_argument("--height_scale_cm", type=float, default=HEIGHT_SCALE_CM)
    parser.add_argument("--min_area_percent", type=float, default=1.5)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    results, meta = estimate_weights(
        args.image, args.seg_weights, args.plate_diameter_cm, args.height_scale_cm,
        args.min_area_percent, args.min_confidence,
    )

    if meta is None:
        print("No items detected.")
        return

    print(f"\nPlate scale: {'auto-detected' if meta['plate_detected'] else 'FALLBACK (no circle found)'} "
          f"| assumed diameter={meta['plate_diameter_cm_assumed']}cm | scale={meta['scale_cm_per_px']:.4f} cm/px")
    print("\nEstimated items:")
    for r in results:
        if r["weight_g"] is None:
            print(f"  {r['name']:22s} | area={r['area_cm2']:6.1f}cm^2 | weight=N/A ({r['note']})")
        else:
            print(f"  {r['name']:22s} | area={r['area_cm2']:6.1f}cm^2 | weight={r['weight_g']:6.1f}g")

    total = sum(r["weight_g"] for r in results if r["weight_g"] is not None)
    print(f"\nTotal estimated plate weight: {total:.1f}g")


if __name__ == "__main__":
    main()