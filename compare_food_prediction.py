from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# -----------------------------
# Settings
# -----------------------------
SIZE = 256
NUM_CLASSES = 42
WEIGHTS = "checkpoints/food_seg_multiclass.pth"

ROOT = Path("ITD")

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# -----------------------------
# Class names
# -----------------------------
classes = {
    0: "background",
    1: "Bottle-gourd-curry",
    2: "aloo-capsicum",
    3: "aloo-curry",
    4: "aloo-fry",
    5: "beans-curry",
    6: "beetroot-kobari",
    7: "beetroot-poriyal",
    8: "bisi-bele-bath",
    9: "boondi",
    10: "cabbage-dry",
    11: "channa-brinjal",
    12: "chicken-dum-biryani",
    13: "chutney",
    14: "curd",
    15: "dondakaya-fry",
    16: "kakarakaya-fry",
    17: "kofta-curry",
    18: "leaf-dal",
    19: "mango-pickle",
    20: "masoor-dal",
    21: "mirchi-ka-salan",
    22: "muddha-pappu",
    23: "non-spicy-curry",
    24: "non-spicy-dal",
    25: "pachi-pulusu",
    26: "papad",
    27: "payasam",
    28: "phulka",
    29: "raita",
    30: "rajma",
    31: "rasam",
    32: "salad",
    33: "sambar",
    34: "steamed-rice",
    35: "tomato-pappu",
    36: "veg-dum-briyani",
    37: "veg-pulao",
    38: "Watermelon",
    39: "Papaya",
    40: "Banana",
    41: "Muskmelon",
}

# -----------------------------
# Load model
# -----------------------------
model = deeplabv3_resnet50(
    weights=None,
    weights_backbone="DEFAULT",
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        WEIGHTS,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Loaded:", WEIGHTS)
print("Device:", device)

# -----------------------------
# Select test image
# -----------------------------
image_path = input("\nEnter TEST image path: ").strip()
image_path = Path(image_path)

# Ground-truth mask path
mask_path = ROOT / "test" / "masks" / (
    image_path.stem.replace("_leftImg8bit", "")
    + "_gtFine_labelIds.png"
)

if not mask_path.exists():
    print("\nERROR: Ground-truth mask not found:")
    print(mask_path)
    raise SystemExit

print("Ground truth:", mask_path)

# -----------------------------
# Load image and ground truth
# -----------------------------
original = Image.open(image_path).convert("RGB")

image = original.resize(
    (SIZE, SIZE),
    Image.Resampling.BILINEAR
)

ground_truth = Image.open(mask_path).resize(
    (SIZE, SIZE),
    Image.Resampling.NEAREST
)

gt = np.asarray(
    ground_truth,
    dtype=np.int64
)

# Ignore unknown IDs 42-50
gt[gt >= NUM_CLASSES] = 255

# -----------------------------
# Prepare input
# -----------------------------
x = np.asarray(
    image,
    dtype=np.float32
) / 255.0

x = torch.from_numpy(x).permute(2, 0, 1)
x = x.unsqueeze(0).to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    output = model(x)["out"]

    prediction = torch.argmax(
        output,
        dim=1
    )[0].cpu().numpy()

# -----------------------------
# Calculate IoU
# -----------------------------
print("\n" + "=" * 60)
print("GROUND TRUTH vs PREDICTION")
print("=" * 60)

present_classes = np.unique(gt)
present_classes = present_classes[present_classes != 255]

ious = []

for class_id in present_classes:

    class_id = int(class_id)

    gt_mask = gt == class_id
    pred_mask = prediction == class_id

    intersection = np.logical_and(
        gt_mask,
        pred_mask
    ).sum()

    union = np.logical_or(
        gt_mask,
        pred_mask
    ).sum()

    if union == 0:
        iou = float("nan")
    else:
        iou = intersection / union

    ious.append(iou)

    gt_pixels = gt_mask.sum()
    pred_pixels = pred_mask.sum()

    print(
        f"{class_id:2d} | "
        f"{classes.get(class_id, 'UNKNOWN'):22s} | "
        f"GT pixels: {gt_pixels:6d} | "
        f"Pred pixels: {pred_pixels:6d} | "
        f"IoU: {iou * 100:6.2f}%"
    )

# -----------------------------
# Mean IoU for classes present
# -----------------------------
if ious:
    mean_iou = np.nanmean(ious)
else:
    mean_iou = 0

print("-" * 60)
print(
    f"Mean IoU of classes present: "
    f"{mean_iou * 100:.2f}%"
)

# -----------------------------
# Create consistent colors
# -----------------------------
rng = np.random.default_rng(42)

palette = np.zeros(
    (NUM_CLASSES, 3),
    dtype=np.uint8
)

palette[1:] = rng.integers(
    40,
    255,
    size=(NUM_CLASSES - 1, 3),
    dtype=np.uint8
)

# Unknown / ignored pixels = black
gt_visual = gt.copy()
gt_visual[gt_visual == 255] = 0

gt_color = palette[gt_visual]
pred_color = palette[prediction]

gt_image = Image.fromarray(gt_color).resize(
    original.size,
    Image.Resampling.NEAREST
)

pred_image = Image.fromarray(pred_color).resize(
    original.size,
    Image.Resampling.NEAREST
)

# -----------------------------
# Create overlays
# -----------------------------
gt_overlay = Image.blend(
    original,
    gt_image,
    alpha=0.45
)

pred_overlay = Image.blend(
    original,
    pred_image,
    alpha=0.45
)

gt_overlay.save("ground_truth_overlay.jpg")
pred_overlay.save("prediction_overlay.jpg")

# -----------------------------
# Side-by-side comparison
# -----------------------------
w, h = original.size

comparison = Image.new(
    "RGB",
    (w * 3, h)
)

comparison.paste(original, (0, 0))
comparison.paste(gt_overlay, (w, 0))
comparison.paste(pred_overlay, (w * 2, 0))

# Labels
draw = ImageDraw.Draw(comparison)

draw.text((20, 20), "ORIGINAL", fill="white")
draw.text((w + 20, 20), "GROUND TRUTH", fill="white")
draw.text((w * 2 + 20, 20), "PREDICTION", fill="white")

comparison.save("food_comparison.jpg")

print("\nSaved:")
print("  ground_truth_overlay.jpg")
print("  prediction_overlay.jpg")
print("  food_comparison.jpg")