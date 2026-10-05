"""
utils.py
--------
Multi-label evaluation metrics and training utilities shared by train.py.
"""

import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    hamming_loss,
    average_precision_score,
)


class EarlyStopping:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if `score` is the new best."""
        is_better = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score + self.min_delta)
            or (self.mode == "min" and score < self.best_score - self.min_delta)
        )
        if is_better:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def compute_multilabel_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], threshold: float = 0.5
) -> Dict:
    """
    y_true: (N, C) binary ground-truth array
    y_probs: (N, C) predicted probabilities (post-sigmoid)
    """
    y_pred = (y_probs >= threshold).astype(int)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    h_loss = hamming_loss(y_true, y_pred)

    # mAP: average precision per class, then mean across classes present in y_true.
    ap_per_class = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            ap = average_precision_score(y_true[:, c], y_probs[:, c])
        else:
            ap = np.nan
        ap_per_class.append(ap)
    mAP = float(np.nanmean(ap_per_class))

    per_class = {
        class_names[i]: {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support_per_class[i]),
            "AP": float(ap_per_class[i]) if not np.isnan(ap_per_class[i]) else None,
        }
        for i in range(len(class_names))
    }

    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "hamming_loss": float(h_loss),
        "mAP": mAP,
        "per_class": per_class,
    }


def save_checkpoint(model, optimizer, epoch, class_names, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, model, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
