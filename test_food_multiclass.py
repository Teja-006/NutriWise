from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50


# -----------------------------
# Settings
# -----------------------------
ROOT = Path("ITD")
SIZE = 256
NUM_CLASSES = 42
BATCH_SIZE = 2

WEIGHTS = "checkpoints/food_seg_multiclass_v2.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Dataset
# -----------------------------
class FoodDataset(Dataset):

    def __init__(self):
        self.img_dir = ROOT / "test" / "images"
        self.mask_dir = ROOT / "test" / "masks"

        self.images = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path = self.images[idx]

        mask_path = self.mask_dir / (
            path.stem.replace("_leftImg8bit", "")
            + "_gtFine_labelIds.png"
        )

        image = Image.open(path).convert("RGB").resize(
            (SIZE, SIZE),
            Image.Resampling.BILINEAR
        )

        mask = Image.open(mask_path).resize(
            (SIZE, SIZE),
            Image.Resampling.NEAREST
        )

        x = np.asarray(image, dtype=np.float32) / 255.0
        y = np.asarray(mask, dtype=np.int64)

        # Ignore unknown IDs 42-50
        y[y >= NUM_CLASSES] = 255

        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y.copy())

        return x, y


# -----------------------------
# Dataset / Loader
# -----------------------------
dataset = FoodDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(dataset))


# -----------------------------
# Model
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


# -----------------------------
# Metrics
# -----------------------------
confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

total_correct = 0
total_valid = 0


# -----------------------------
# Evaluation
# -----------------------------
print("\nTesting model...\n")

with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(loader, 1):

        images = images.to(device)

        outputs = model(images)["out"]

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        masks = masks.numpy()

        for pred, true in zip(predictions, masks):

            valid = true != 255

            true_valid = true[valid]
            pred_valid = pred[valid]

            total_correct += np.sum(
                pred_valid == true_valid
            )

            total_valid += len(true_valid)

            # Confusion matrix
            for t, p in zip(true_valid, pred_valid):
                confusion[t, p] += 1

        if batch_idx % 50 == 0:
            print(
                f"Processed "
                f"{min(batch_idx * BATCH_SIZE, len(dataset))}"
                f"/{len(dataset)}"
            )


# -----------------------------
# Pixel Accuracy
# -----------------------------
pixel_accuracy = (
    total_correct / total_valid
    if total_valid > 0 else 0
)


# -----------------------------
# IoU
# -----------------------------
ious = []

for class_id in range(NUM_CLASSES):

    intersection = confusion[class_id, class_id]

    ground_truth = confusion[class_id, :].sum()

    predicted = confusion[:, class_id].sum()

    union = ground_truth + predicted - intersection

    if union > 0:
        iou = intersection / union
        ious.append(iou)
    else:
        iou = float("nan")

    print(
        f"Class {class_id:2d} | "
        f"IoU: {iou:.4f}"
    )


mean_iou = np.nanmean(ious)


# -----------------------------
# Results
# -----------------------------
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

print(f"Pixel Accuracy : {pixel_accuracy * 100:.2f}%")
print(f"Mean IoU       : {mean_iou * 100:.2f}%")
print(f"Valid pixels   : {total_valid}")

print("=" * 50)