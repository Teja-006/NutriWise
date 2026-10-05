from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# -----------------------------
# Settings
# -----------------------------
SIZE = 256
NUM_CLASSES = 42
WEIGHTS = "checkpoints/food_seg_multiclass_v2.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Same class mapping used by the project
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

print("Loaded weights:", WEIGHTS)
print("Device:", device)

# -----------------------------
# Get image
# -----------------------------
image_path = input("\nEnter image path: ").strip()

image = Image.open(image_path).convert("RGB")

original = image.copy()

image_resized = image.resize(
    (SIZE, SIZE),
    Image.Resampling.BILINEAR
)

x = np.asarray(
    image_resized,
    dtype=np.float32
) / 255.0

x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
x = x.to(device)

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
# Find detected classes
# -----------------------------
detected = np.unique(prediction)

print("\nDetected classes:")

for class_id in detected:

    pixel_count = np.sum(prediction == class_id)

    percentage = (
        pixel_count / prediction.size
    ) * 100

    print(
        f"{class_id:2d} | "
        f"{classes.get(class_id, 'UNKNOWN')} | "
        f"{percentage:.2f}% of image"
    )

# -----------------------------
# Create color mask
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

color_mask = palette[prediction]

mask_image = Image.fromarray(
    color_mask
).resize(
    original.size,
    Image.Resampling.NEAREST
)

# -----------------------------
# Overlay
# -----------------------------
overlay = Image.blend(
    original,
    mask_image,
    alpha=0.45
)

mask_image.save("multiclass_mask.png")
overlay.save("multiclass_overlay.jpg")

print("\nSaved:")
print("  multiclass_mask.png")
print("  multiclass_overlay.jpg")