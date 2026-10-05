from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

ROOT = Path("ITD")
SIZE = 256
BATCH_SIZE = 2
EPOCHS = 3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

class FoodDataset(Dataset):
    def __init__(self, split):
        self.img_dir = ROOT / split / "images"
        self.mask_dir = ROOT / split / "masks"
        self.images = sorted(self.img_dir.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No JPG images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        mask_path = self.mask_dir / (path.stem.replace("_leftImg8bit", "") + "_gtFine_labelIds.png")

        image = Image.open(path).convert("RGB").resize((SIZE, SIZE))
        mask = Image.open(mask_path).resize((SIZE, SIZE), Image.Resampling.NEAREST)

        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask) != 0).astype(np.int64)

        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y.copy())
        return x, y

train_ds = FoodDataset("train")
val_ds = FoodDataset("test")

# Small initial run: use a subset so we can verify the pipeline quickly.
rng = random.Random(42)
train_ids = rng.sample(range(len(train_ds)), min(256, len(train_ds)))
val_ids = rng.sample(range(len(val_ds)), min(64, len(val_ds)))

train_loader = DataLoader(
    torch.utils.data.Subset(train_ds, train_ids),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    torch.utils.data.Subset(val_ds, val_ids),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, (images, masks) in enumerate(train_loader, 1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)["out"]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct = pixels = 0
    with torch.no_grad():
        for images, masks in val_loader:
            logits = model(images.to(device))["out"]
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == masks).sum().item()
            pixels += masks.numel()

    print(f"Validation pixel accuracy: {correct / pixels:.4f}")

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/food_seg_binary.pth")
print("Saved: checkpoints/food_seg_binary.pth")
