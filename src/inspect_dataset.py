"""
inspect_dataset.py
-------------------
RUN THIS FIRST, before touching dataset.py.

This script does NOT assume any folder structure. It scans whatever you
place under `data/` and prints a report of:
  - directory tree (top few levels)
  - image file counts / extensions found
  - any CSV / JSON / XML / TXT annotation files it finds, plus a preview
    of their contents (columns, keys, first few rows/entries)

Use the printed report to decide which loader branch in dataset.py
(`_load_csv_format`, `_load_json_coco_format`, or `_load_folder_format`)
matches your actual dataset, and adjust `DATASET_FORMAT` in train.py's
config accordingly.

Usage:
    python src/inspect_dataset.py --data_dir data
"""

import argparse
import json
import os
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTS = {".csv", ".json", ".xml", ".txt"}
MASK_DIR_HINTS = {"mask", "masks", "gt", "gtfine", "annotations", "labels", "seg", "segmentation"}
CLASSMAP_NAME_HINTS = {"class", "classes", "label", "labels", "categories", "meta", "colormap", "color_map"}


def print_tree(root, max_depth=3, max_entries_per_dir=15):
    print(f"\n=== Directory tree (root: {root}) ===")
    root = os.path.abspath(root)
    root_depth = root.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []  # stop descending
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/")
        shown = filenames[:max_entries_per_dir]
        for f in shown:
            print(f"{indent}  {f}")
        if len(filenames) > max_entries_per_dir:
            print(f"{indent}  ... (+{len(filenames) - max_entries_per_dir} more files)")


def scan_files(root):
    image_ext_counts = Counter()
    annotation_files = []
    total_images = 0

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                image_ext_counts[ext] += 1
                total_images += 1
            elif ext in ANNOTATION_EXTS:
                annotation_files.append(os.path.join(dirpath, f))

    print("\n=== Image files ===")
    print(f"Total images found: {total_images}")
    for ext, count in image_ext_counts.items():
        print(f"  {ext}: {count}")

    print("\n=== Candidate annotation files ===")
    if not annotation_files:
        print("  None found. If your dataset uses folder-per-class structure "
              "(e.g. data/train/Rice/img1.jpg, data/train/Dal/img1.jpg with "
              "shared filenames across class folders indicating co-occurrence), "
              "dataset.py's folder-based loader can be adapted for that.")
    for path in annotation_files:
        print(f"  {path}")

    return annotation_files


def preview_annotation_file(path, max_chars=2000):
    ext = os.path.splitext(path)[1].lower()
    print(f"\n--- Preview: {path} ---")
    try:
        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"Top-level keys: {list(data.keys())}")
                for k, v in data.items():
                    if isinstance(v, list) and v:
                        print(f"  '{k}' is a list of {len(v)} items. First item:")
                        print(f"    {json.dumps(v[0], indent=2)[:max_chars]}")
            elif isinstance(data, list) and data:
                print(f"Top-level list of {len(data)} items. First item:")
                print(f"  {json.dumps(data[0], indent=2)[:max_chars]}")
        elif ext == ".csv":
            with open(path, "r") as f:
                lines = [next(f) for _ in range(6) if f]
            print("First 6 lines:")
            for line in lines:
                print(f"  {line.rstrip()}")
        elif ext == ".txt":
            with open(path, "r") as f:
                lines = [next(f) for _ in range(6) if f]
            print("First 6 lines:")
            for line in lines:
                print(f"  {line.rstrip()}")
        elif ext == ".xml":
            with open(path, "r") as f:
                content = f.read(max_chars)
            print(content)
    except StopIteration:
        pass
    except Exception as e:
        print(f"  Could not preview file: {e}")


def inspect_mask_folders(root):
    """
    Finds folders that look like mask/label directories, samples a few
    files from each, and reports:
      - image mode (e.g. 'L' = single-channel index mask, 'P' = palette,
        'RGB'/'RGBA' = color-coded mask)
      - the set of unique pixel values / colors present (capped) so we can
        tell whether masks encode class-index-per-pixel or color-per-class
      - whether mask filenames correspond 1:1 with an images folder by stem
    """
    from PIL import Image
    import numpy as np

    print("\n=== Mask / label folder inspection ===")
    mask_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base in MASK_DIR_HINTS:
            mask_dirs.append(dirpath)

    if not mask_dirs:
        print("  No folder matched common mask-directory names "
              f"({sorted(MASK_DIR_HINTS)}). If your masks live elsewhere, "
              "pass --mask_dir explicitly.")
        return

    for mdir in mask_dirs:
        files = sorted([f for f in os.listdir(mdir)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
        print(f"\n--- Mask folder: {mdir} ({len(files)} files) ---")
        if not files:
            continue

        sample_files = files[:3]
        for fname in sample_files:
            fpath = os.path.join(mdir, fname)
            try:
                img = Image.open(fpath)
                arr = np.array(img)
                print(f"  {fname}")
                print(f"    mode={img.mode}, size={img.size}, array_shape={arr.shape}, dtype={arr.dtype}")
                if img.mode in ("L", "P", "I"):
                    uniq = np.unique(arr)
                    capped = uniq[:30]
                    print(f"    unique pixel values ({len(uniq)} total): {capped.tolist()}"
                          f"{' ...' if len(uniq) > 30 else ''}")
                elif img.mode in ("RGB", "RGBA"):
                    flat = arr.reshape(-1, arr.shape[-1])
                    uniq_colors = np.unique(flat, axis=0)
                    capped = uniq_colors[:20]
                    print(f"    unique colors ({len(uniq_colors)} total): {capped.tolist()}"
                          f"{' ...' if len(uniq_colors) > 20 else ''}")
            except Exception as e:
                print(f"    Could not read '{fpath}': {e}")

        # Check filename correspondence with a sibling 'images' folder
        parent = os.path.dirname(mdir)
        images_dir = os.path.join(parent, "images")
        if os.path.isdir(images_dir):
            img_stems = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
            mask_stems = {os.path.splitext(f)[0] for f in files}
            overlap = img_stems & mask_stems
            print(f"\n  Sibling 'images' folder found: {images_dir}")
            print(f"  Exact filename-stem matches with masks: {len(overlap)} / {len(mask_stems)} masks")
            if len(overlap) < len(mask_stems) * 0.5:
                sample_img_stem = next(iter(img_stems)) if img_stems else "N/A"
                sample_mask_stem = next(iter(mask_stems)) if mask_stems else "N/A"
                print(f"  Low overlap — filenames likely differ by a suffix pattern. "
                      f"Example image stem: '{sample_img_stem}', example mask stem: '{sample_mask_stem}'")


def find_classmap_candidates(root):
    print("\n=== Class-mapping file candidates ===")
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            stem = os.path.splitext(f)[0].lower()
            ext = os.path.splitext(f)[1].lower()
            if ext in {".csv", ".json", ".txt"} and any(h in stem for h in CLASSMAP_NAME_HINTS):
                candidates.append(os.path.join(dirpath, f))
    if not candidates:
        print("  None found by filename heuristics. Check the ITD dataset's "
              "README/paper for the class list — segmentation datasets almost "
              "always ship one (often named differently, e.g. 'dish_ids.json').")
    for c in candidates:
        print(f"  {c}")
        preview_annotation_file(c)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Inspect an Indian food dataset's structure.")
    parser.add_argument("--data_dir", type=str, default="data",
                         help="Path to the dataset root directory.")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--preview_annotations", action="store_true", default=True,
                         help="Print a preview of each annotation file found.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"'{args.data_dir}' does not exist. Place your dataset there first, "
            f"e.g. NutriWise/data/<your dataset files>."
        )

    print_tree(args.data_dir, max_depth=args.max_depth)
    annotation_files = scan_files(args.data_dir)

    if args.preview_annotations:
        for path in annotation_files[:10]:  # cap to avoid flooding output
            preview_annotation_file(path)

    inspect_mask_folders(args.data_dir)
    find_classmap_candidates(args.data_dir)

    print("\n=== Next step ===")
    print("If a 'Mask / label folder' section printed above with unique pixel")
    print("values or colors, and a class-mapping file was found, this is a")
    print("segmentation-style dataset. Share this report so the loader can be")
    print("written for your exact mask encoding (index-per-pixel vs RGB-per-class)")
    print("and class-id mapping.")
    print("Otherwise, open src/dataset.py and:")
    print("  1. Set --dataset_format to 'csv', 'coco_json', or 'folder'.")
    print("  2. If none of the built-in loaders match, edit the matching")
    print("     `_load_*_format` method in dataset.py to parse your specific")
    print("     annotation schema into the internal (image_path, plate_id, [class_names]) format.")


if __name__ == "__main__":
    main()