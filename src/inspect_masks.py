from pathlib import Path
import numpy as np
from PIL import Image


MASK_DIR = Path("data/train/masks")

all_values = set()

mask_files = list(MASK_DIR.glob("*.png"))

print(f"Total masks: {len(mask_files)}\n")


for i, mask_path in enumerate(mask_files):

    mask = Image.open(mask_path)

    mask = np.array(mask)

    unique_values = np.unique(mask)

    all_values.update(unique_values.tolist())

    # Show first 10 masks
    if i < 10:
        print(f"Mask: {mask_path.name}")
        print("Unique values:", unique_values)

    # Progress every 100 masks
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(mask_files)} masks...")


print("\n==========================")
print("ALL UNIQUE LABEL VALUES")
print("==========================")

print(sorted(all_values))