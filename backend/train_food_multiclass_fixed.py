"""
train_food_multiclass_fixed.py
-------------------------------
Fixed version of train_food_multiclass.py.

What changed vs. the original, and why:

1. BATCH_SIZE was 2. DeepLabV3's ResNet50 backbone relies heavily on
   BatchNorm, which needs a reasonably large batch to compute stable
   running statistics. Batch size 2 gives BatchNorm near-garbage stats,
   which makes predictions noisy at inference time. Default here is 8,
   with --accum_steps to simulate a bigger effective batch
   (effective_batch = batch_size * accum_steps) if GPU memory is tight.

2. No class weighting. With 42 classes where background/rice dominate
   pixel counts, rare dishes get drowned out. This script computes
   inverse-frequency class weights from the training masks and passes
   them into CrossEntropyLoss.

3. No ImageNet normalization, despite using an ImageNet-pretrained
   backbone (weights_backbone="DEFAULT"). Added standard mean/std
   normalization (can be disabled with --no_normalize to reproduce the
   exact old behavior).

4. Validation == test set. The original script picked its "best"
   checkpoint using the test set, which makes any reported accuracy
   optimistic. This script carves a held-out val split OUT OF the
   train folder (--val_split, default 0.1) and never touches "test" —
   keep using test_food_multiclass.py on the test folder for the real
   final number.

5. No GPU support beyond Apple "mps". Added "cuda" detection first
   (so this trains properly on Colab/any NVIDIA GPU), mps second,
   cpu last.

6. Metric for checkpoint selection changed from val loss to val mean
   IoU — closer to what you actually care about (did it find the
   right regions), and matches how test_food_multiclass.py already
   scores the model.

7. Checkpoints now save a metadata dict (state_dict + num_classes +
   img_size + normalize flag) instead of a bare state_dict, so
   predict_food_multiclass_fixed.py can auto-configure itself instead
   of you having to remember what settings a checkpoint was trained
   with.

Usage (same dataset layout as before: ITD/train/{images,masks},
ITD/test/{images,masks}, Cityscapes-style mask filenames):

    python train_food_multiclass_fixed.py --epochs 30 --batch_size 8

On a small/shared GPU, keep batch_size lower and raise accum_steps to get
the same effective batch size, e.g. --batch_size 4 --accum_steps 2.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50

from food_classes import NUM_CLASSES as DEFAULT_NUM_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 food segmentation (fixed)")
    parser.add_argument("--data_dir", type=str, default="ITD")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="food_seg_multiclass_v3.pth")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=1,
                         help="Gradient accumulation steps; effective batch = batch_size * accum_steps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1,
                         help="Fraction of the TRAIN folder held out for validation (test folder is never touched here)")
    parser.add_argument("--no_normalize", action="store_true",
                         help="Disable ImageNet normalization (reproduces original script's behavior)")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable random horizontal flip augmentation on the train split")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU. Training will be slow.")
    return device


# ============================================================
# DATASET
# ============================================================
class FoodSegDataset(Dataset):
    """
    Same Cityscapes-style mask convention as the original script:
    <stem>_gtFine_labelIds.png next to <stem>_leftImg8bit.jpg (or plain
    <stem>.jpg — the replace() is a no-op if that suffix isn't present).
    """

    def __init__(self, img_dir, mask_dir, img_size, num_classes,
                 normalize=True, augment=False):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(self.img_dir.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No JPG images found in {self.img_dir}")
        self.img_size = img_size
        self.num_classes = num_classes
        self.normalize = normalize
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        mask_path = self.mask_dir / (
            path.stem.replace("_leftImg8bit", "") + "_gtFine_labelIds.png"
        )

        image = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).resize(
            (self.img_size, self.img_size), Image.Resampling.NEAREST
        )

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        x = np.asarray(image, dtype=np.float32) / 255.0
        if self.normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = np.asarray(mask, dtype=np.int64)
        y[y >= self.num_classes] = 255  # unknown IDs -> ignored in loss

        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y.copy())
        return x, y


# ============================================================
# CLASS WEIGHTS (inverse pixel frequency, computed from TRAIN split only)
# ============================================================
def compute_class_weights(dataset, indices, num_classes, max_samples=300):
    """
    Scans up to `max_samples` masks (randomly sampled from the train split)
    and counts pixels per class. Rare classes get a higher weight so the
    loss doesn't just learn to always predict background/rice.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    sample_indices = indices if len(indices) <= max_samples else random.sample(indices, max_samples)

    for idx in sample_indices:
        _, mask_path_idx = dataset.images[idx], None
        mask_path = dataset.mask_dir / (
            dataset.images[idx].stem.replace("_leftImg8bit", "") + "_gtFine_labelIds.png"
        )
        mask = np.asarray(Image.open(mask_path))
        mask = mask[mask < num_classes]
        class_ids, class_counts = np.unique(mask, return_counts=True)
        counts[class_ids] += class_counts

    counts = np.clip(counts, 1, None)  # avoid div-by-zero for classes never sampled
    freq = counts / counts.sum()
    weights = 1.0 / np.log(1.02 + freq)  # smoothed inverse frequency, avoids extreme outliers
    weights = weights / weights.mean()   # normalize so average weight ~1
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# METRICS (pixel accuracy + mean IoU, same definitions as test_food_multiclass.py)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct, total_valid, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)["out"]
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        masks_np = masks.cpu().numpy()

        for pred, true in zip(preds, masks_np):
            valid = true != 255
            t, p = true[valid], pred[valid]
            total_correct += np.sum(p == t)
            total_valid += len(t)
            for ti, pi in zip(t, p):
                confusion[ti, pi] += 1

    pixel_acc = total_correct / total_valid if total_valid else 0.0

    ious = []
    for c in range(num_classes):
        inter = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - inter
        if union > 0:
            ious.append(inter / union)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "loss": total_loss / len(loader.dataset),
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    root = Path(args.data_dir)
    normalize = not args.no_normalize

    # ---- Build full train-folder dataset, then split into train/val ----
    full_train_ds = FoodSegDataset(
        root / "train" / "images", root / "train" / "masks",
        args.img_size, args.num_classes, normalize=normalize, augment=not args.no_augment,
    )
    # A second instance with augment=False, reused for the val subset so
    # validation images are never flipped/augmented.
    full_train_ds_noaug = FoodSegDataset(
        root / "train" / "images", root / "train" / "masks",
        args.img_size, args.num_classes, normalize=normalize, augment=False,
    )

    n = len(full_train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = max(1, int(n * args.val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds_noaug, val_indices)

    test_ds = FoodSegDataset(
        root / "test" / "images", root / "test" / "masks",
        args.img_size, args.num_classes, normalize=normalize, augment=False,
    )

    print(f"Train: {len(train_ds)} | Val (held out from train): {len(val_ds)} | Test (untouched): {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ---- Class weights ----
    if args.no_class_weights:
        class_weights = None
        print("Class weighting disabled.")
    else:
        print("Computing class weights from train split...")
        class_weights = compute_class_weights(full_train_ds, train_indices, args.num_classes).to(device)
        print(f"Class weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")

    # ---- Model ----
    model = deeplabv3_resnet50(weights=None, weights_backbone="DEFAULT", num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    ckpt_path = Path(args.checkpoint_dir) / args.checkpoint_name

    best_val_iou = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)["out"]
            loss = criterion(logits, masks) / args.accum_steps
            loss.backward()

            if step % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_steps

            if step % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {step}/{len(train_loader)} | Loss {loss.item() * args.accum_steps:.4f}")

        # flush any leftover accumulated gradients
        if len(train_loader) % args.accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, args.num_classes)
        scheduler.step(val_metrics["mean_iou"])

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} COMPLETE | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_pixel_acc={val_metrics['pixel_acc']:.4f} | val_mean_iou={val_metrics['mean_iou']:.4f}\n"
            + "-" * 60
        )

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "img_size": args.img_size,
                "normalize": normalize,
            }, ckpt_path)
            print(f"Saved new best checkpoint ({ckpt_path}) | val_mean_iou={best_val_iou:.4f}\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best val mean IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    # ---- Final, real held-out test number (never used for model selection) ----
    print("\nEvaluating best checkpoint on the held-out TEST set...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.num_classes)
    print(f"TEST pixel_acc={test_metrics['pixel_acc']:.4f} | TEST mean_iou={test_metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()