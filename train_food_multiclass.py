from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path("ITD")

SIZE = 256
BATCH_SIZE = 2
EPOCHS = 30
NUM_CLASSES = 42

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

class FoodDataset(Dataset):

    def __init__(self, split):

        self.img_dir = ROOT / split / "images"
        self.mask_dir = ROOT / split / "masks"

        self.images = sorted(
            self.img_dir.glob("*.jpg")
        )

        print(
            f"{split} dataset: "
            f"{len(self.images)} images"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path = self.images[idx]

        mask_path = self.mask_dir / (
            path.stem.replace(
                "_leftImg8bit",
                ""
            )
            + "_gtFine_labelIds.png"
        )

        image = Image.open(
            path
        ).convert("RGB").resize(
            (SIZE, SIZE),
            Image.Resampling.BILINEAR
        )

        mask = Image.open(
            mask_path
        ).resize(
            (SIZE, SIZE),
            Image.Resampling.NEAREST
        )

        # Convert image to tensor
        x = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Convert mask to integer class IDs
        y = np.asarray(
            mask,
            dtype=np.int64
        )

        # Unknown IDs 42-50:
        # ignore them instead of treating them as background
        y[y >= NUM_CLASSES] = 255

        x = torch.from_numpy(
            x
        ).permute(2, 0, 1)

        y = torch.from_numpy(
            y.copy()
        )

        return x, y


# ============================================================
# LOAD DATA
# ============================================================

train_ds = FoodDataset("train")
test_ds = FoodDataset("test")

# Use ALL training images
train_ids = list(
    range(len(train_ds))
)

# Use ALL test images
test_ids = list(
    range(len(test_ds))
)


train_loader = DataLoader(
    torch.utils.data.Subset(
        train_ds,
        train_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)


test_loader = DataLoader(
    torch.utils.data.Subset(
        test_ds,
        test_ids
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=True
)


print()
print("Training images:", len(train_ids))
print("Test images:", len(test_ids))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Number of classes:", NUM_CLASSES)


# ============================================================
# MODEL
# ============================================================

model = deeplabv3_resnet50(
    weights=None,
    weights_backbone="DEFAULT",
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

Path("checkpoints").mkdir(
    exist_ok=True
)


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    total_train_loss = 0.0

    for step, (images, masks) in enumerate(
        train_loader,
        1
    ):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)["out"]

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        if step % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)["out"]

            loss = criterion(
                logits,
                masks
            )

            total_val_loss += loss.item()

    val_loss = (
        total_val_loss /
        len(test_loader)
    )


    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} COMPLETE"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print("-" * 60)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/"
            "food_seg_multiclass_v2.pth"
        )

        print(
            "✓ Saved best model:"
        )

        print(
            "  checkpoints/"
            "food_seg_multiclass_v2.pth"
        )

        print(
            f"  Best validation loss: "
            f"{best_val_loss:.4f}"
        )

        print("-" * 60)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Best weights:"
)

print(
    "checkpoints/"
    "food_seg_multiclass_v2.pth"
)