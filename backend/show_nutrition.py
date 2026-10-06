"""
show_nutrition.py
-----------------
Runs weight estimation, then prints grams + calories, protein, carbs,
fat and fiber for each item and the whole plate.

Usage:
    python show_nutrition.py
        (it asks you for the image path)
    python show_nutrition.py --image path/to/thali.jpg
        (skips the question)
"""

import argparse
import os

from estimate_weight import estimate_weights, PLATE_DIAMETER_CM, HEIGHT_SCALE_CM
from food_nutrition import nutrition_for, FIELDS


def ask_image_path():
    while True:
        raw = input("\nEnter image path (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        # clean up drag-and-drop paths: quotes, escaped spaces, ~
        path = raw.strip("'\"").replace("\\ ", " ")
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            return path
        print(f"File not found: {path}\nTry again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="optional; if omitted you are asked")
    parser.add_argument("--seg_weights", default="checkpoints/food_seg_multiclass_v3.pth")
    parser.add_argument("--plate_diameter_cm", type=float, default=PLATE_DIAMETER_CM)
    parser.add_argument("--height_scale_cm", type=float, default=HEIGHT_SCALE_CM)
    parser.add_argument("--min_area_percent", type=float, default=1.5)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    image_path = args.image if args.image else ask_image_path()
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return

    print("\nRunning... (the first run may download the depth model)")
    results, meta = estimate_weights(
        image_path, args.seg_weights, args.plate_diameter_cm, args.height_scale_cm,
        args.min_area_percent, args.min_confidence,
    )

    if meta is None:
        print("No items detected.")
        return

    print(f"\nPlate scale: {'auto-detected' if meta['plate_detected'] else 'FALLBACK (no circle found)'}")
    print(f"\n{'Item':30s} {'grams':>7s} {'kcal':>7s} {'protein':>8s} {'carbs':>7s} {'fat':>6s} {'fiber':>6s}")
    print("-" * 80)

    totals = {k: 0.0 for k in FIELDS}
    total_g = 0.0
    missing = []

    for r in results:
        g = r["weight_g"]
        if g is None:
            print(f"{r['name']:30s} {'N/A':>7s}   (no weight data)")
            continue
        total_g += g
        n = nutrition_for(r["name"], g)
        if n is None:
            missing.append(r["name"])
            print(f"{r['name']:30s} {g:7.1f} {'N/A':>7s}   (no nutrition data)")
            continue
        for k in FIELDS:
            totals[k] += n[k]
        print(f"{r['name']:30s} {g:7.1f} {n['calories_kcal']:7.1f} {n['protein_g']:7.1f}g "
              f"{n['carbs_g']:6.1f}g {n['fat_g']:5.1f}g {n['fiber_g']:5.1f}g")

    print("-" * 80)
    print(f"{'TOTAL':30s} {total_g:7.1f} {totals['calories_kcal']:7.1f} {totals['protein_g']:7.1f}g "
          f"{totals['carbs_g']:6.1f}g {totals['fat_g']:5.1f}g {totals['fiber_g']:5.1f}g")

    if missing:
        print(f"\nNote: no nutrition data yet for: {', '.join(missing)} (totals exclude them)")


if __name__ == "__main__":
    main()