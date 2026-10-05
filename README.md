# NutriWise — Multi-Label Indian Food Classification

Detects **all** food items present in a single thali/plate image (e.g. Rice,
Dal, Roti, Paneer at once) using transfer learning on ResNet50, trained as a
**multi-label** (not multi-class) problem.

## Project Structure

```
NutriWise/
├── data/                  # Put your dataset here
├── src/
│   ├── inspect_dataset.py # RUN THIS FIRST — reports your dataset's structure
│   ├── dataset.py         # Multi-label loader + group-aware train/val/test split
│   ├── model.py            # ResNet50 with sigmoid-ready output head
│   ├── train.py            # Two-phase transfer learning + full metric suite
│   ├── predict.py          # Single-image inference
│   └── utils.py            # Metrics (F1 micro/macro, Hamming loss, mAP) + early stopping
├── checkpoints/            # best_model.pt + training_curves.png saved here
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab:
```python
!git clone <your-repo-or-upload-zip>
%cd NutriWise
!pip install -r requirements.txt
```

## Step 1 — Inspect your dataset

Place your Indian Thali Dataset (or similar) inside `data/`, then run:

```bash
python src/inspect_dataset.py --data_dir data
```

This prints the folder tree, image counts, and previews any CSV/JSON/XML
annotation files it finds — no assumptions are made about your dataset's
layout.

## Step 2 — Pick the matching dataset format

Based on the report, choose `--dataset_format`:

| Format | When to use |
|---|---|
| `csv` | An `annotations.csv` with one row per image, a labels column (e.g. `Rice;Dal;Roti`), optionally a `plate_id` column |
| `coco_json` | COCO-style `images` / `annotations` / `categories` JSON, multiple annotations per image aggregated into one multi-hot vector |
| `folder` | `data/<ClassName>/<image>` folders, where images sharing a filename stem across class folders belong to the same plate |

If none match exactly, edit the corresponding `_load_*_format` method in
`src/dataset.py` — everything downstream (splitting, augmentation, training)
only depends on the internal record format:
`{"image_path": ..., "plate_id": ..., "labels": [...]}`.

**Leakage protection:** splitting is done by `plate_id`, via
`GroupShuffleSplit`, so multiple photos of the same physical thali can never
end up in both train and test.

## Step 3 — Train

```bash
python src/train.py \
    --data_dir data \
    --dataset_format csv \
    --phase1_epochs 5 \
    --phase2_epochs 15 \
    --batch_size 32
```

Training runs in two phases:
1. **Phase 1** — ResNet50 backbone frozen, only the new classification head trains (fast, prevents destroying pretrained features early).
2. **Phase 2** — `layer4` (configurable via `--unfreeze_blocks`) is unfrozen and fine-tuned at a lower learning rate.

The best checkpoint (by **validation micro-F1**) is saved to
`checkpoints/best_model.pt`. Training/validation loss and F1 curves are
saved to `checkpoints/training_curves.png`. Early stopping and
`ReduceLROnPlateau` are both active in each phase.

At the end, the script reports on the held-out **test set**:
precision/recall/F1 (micro & macro), Hamming loss, mAP, and a full
per-class breakdown.

## Step 4 — Predict on a new image

```bash
python src/predict.py --image path/to/thali.jpg --checkpoint checkpoints/best_model.pt --threshold 0.5
```

Example output:
```
Per-class probabilities:
  Rice: 0.97
  Dal: 0.92
  Roti: 0.88
  Paneer: 0.21
  Pickle: 0.04

Final detected foods (threshold=0.5):
['Rice', 'Dal', 'Roti']
```

## Design Notes

- **Sigmoid, not Softmax**: each class is an independent binary decision (multiple foods can be simultaneously present), so the output head is `Linear(2048, num_classes)` with `BCEWithLogitsLoss` — sigmoid is applied internally by the loss for numerical stability, and explicitly at inference time.
- **Why micro-F1 for checkpointing, not loss**: with class imbalance (some foods appear far more often than others), validation loss can look fine while rare classes are being ignored. Micro-F1 tracks actual detection quality across all classes.
- **GPU/CPU**: `model.py`'s `get_device()` auto-detects CUDA and falls back to CPU with a warning.
