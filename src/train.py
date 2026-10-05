"""
train.py
--------
Two-phase transfer learning for multi-label Indian food classification.

Phase 1:
Freeze ResNet50 backbone and train only the classification head.

Phase 2:
Unfreeze layer4 and fine-tune at a lower learning rate.

Usage:
    python src/train.py
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import build_datasets
from model import build_model, get_device


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train Multi-Label ResNet50 Food Classifier"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--phase1_lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--phase2_lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# ============================================================
# MULTI-LABEL METRICS
# ============================================================

def calculate_metrics(
    outputs,
    labels,
    threshold=0.5
):

    # Convert logits to probabilities

    probabilities = torch.sigmoid(
        outputs
    )


    # Convert probabilities to predictions

    predictions = (

        probabilities >= threshold

    ).float()


    # Exact match accuracy
    #
    # An image is correct only if ALL
    # predicted labels match the true labels.

    exact_matches = (

        predictions == labels

    ).all(
        dim=1
    ).sum().item()


    # Element-wise label accuracy

    label_correct = (

        predictions == labels

    ).sum().item()


    label_total = labels.numel()


    return (

        exact_matches,

        label_correct,

        label_total

    )


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(

    model,
    loader,
    criterion,
    device,
    optimizer=None,
    threshold=0.5

):

    is_train = optimizer is not None


    if is_train:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    total_images = 0

    total_exact_matches = 0

    total_label_correct = 0

    total_labels = 0


    with torch.set_grad_enabled(

        is_train

    ):


        for images, labels in loader:


            images = images.to(

                device

            )


            labels = labels.to(

                device

            )


            # --------------------
            # Zero Gradients
            # --------------------

            if is_train:

                optimizer.zero_grad()


            # --------------------
            # Forward Pass
            # --------------------

            outputs = model(

                images

            )


            loss = criterion(

                outputs,

                labels

            )


            # --------------------
            # Backpropagation
            # --------------------

            if is_train:


                loss.backward()


                optimizer.step()


            # --------------------
            # Loss
            # --------------------

            batch_size = images.size(

                0

            )


            total_loss += (

                loss.item()

                *

                batch_size

            )


            total_images += batch_size


            # --------------------
            # Metrics
            # --------------------

            (

                exact_matches,

                label_correct,

                label_total

            ) = calculate_metrics(

                outputs,

                labels,

                threshold

            )


            total_exact_matches += (

                exact_matches

            )


            total_label_correct += (

                label_correct

            )


            total_labels += (

                label_total

            )


    average_loss = (

        total_loss

        /

        total_images

    )


    exact_match_accuracy = (

        total_exact_matches

        /

        total_images

    )


    label_accuracy = (

        total_label_correct

        /

        total_labels

    )


    return (

        average_loss,

        exact_match_accuracy,

        label_accuracy

    )


# ============================================================
# TRAINING PHASE
# ============================================================

def train_phase(

    phase_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_path,
    history,
    class_names,
    threshold

):


    print(

        "\n"

        +

        "=" * 60

    )


    print(

        phase_name

    )


    print(

        f"Trainable parameters: "

        f"{model.trainable_parameter_count():,}"

    )


    print(

        "=" * 60

    )


    best_accuracy = 0.0


    for epoch in range(

        1,

        num_epochs + 1

    ):


        start_time = time.time()


        # ====================================================
        # TRAIN
        # ====================================================

        (

            train_loss,

            train_exact_accuracy,

            train_label_accuracy

        ) = run_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            device=device,

            optimizer=optimizer,

            threshold=threshold

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (

            val_loss,

            val_exact_accuracy,

            val_label_accuracy

        ) = run_epoch(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device,

            threshold=threshold

        )


        # Update learning rate

        scheduler.step(

            val_loss

        )


        elapsed = (

            time.time()

            -

            start_time

        )


        print(

            f"[{phase_name}] "

            f"Epoch {epoch}/{num_epochs} "

            f"({elapsed:.1f}s) | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Label Acc: "

            f"{train_label_accuracy:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Label Acc: "

            f"{val_label_accuracy:.4f}"

        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history["train_loss"].append(

            train_loss

        )


        history["val_loss"].append(

            val_loss

        )


        history["train_accuracy"].append(

            train_label_accuracy

        )


        history["val_accuracy"].append(

            val_label_accuracy

        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_label_accuracy > best_accuracy:


            best_accuracy = (

                val_label_accuracy

            )


            torch.save(

                {

                    "model_state_dict":

                        model.state_dict(),


                    "optimizer_state_dict":

                        optimizer.state_dict(),


                    "epoch":

                        epoch,


                    "val_accuracy":

                        val_label_accuracy,


                    "threshold":

                        threshold,


                    "class_names":

                        class_names,


                    "num_classes":

                        len(class_names)

                },

                checkpoint_path

            )


            print(

                f"  -> New best model! "

                f"Validation Label Accuracy: "

                f"{val_label_accuracy:.4f}"

            )


    return best_accuracy


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

def plot_history(

    history,
    output_path

):


    plt.figure(

        figsize=(12, 5)

    )


    # ========================================================
    # LOSS GRAPH
    # ========================================================

    plt.subplot(

        1,

        2,

        1

    )


    plt.plot(

        history["train_loss"],

        label="Training Loss"

    )


    plt.plot(

        history["val_loss"],

        label="Validation Loss"

    )


    plt.title(

        "Training and Validation Loss"

    )


    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Loss"

    )


    plt.legend()


    # ========================================================
    # ACCURACY GRAPH
    # ========================================================

    plt.subplot(

        1,

        2,

        2

    )


    plt.plot(

        history["train_accuracy"],

        label="Training Label Accuracy"

    )


    plt.plot(

        history["val_accuracy"],

        label="Validation Label Accuracy"

    )


    plt.title(

        "Training and Validation Label Accuracy"

    )


    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Accuracy"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        output_path

    )


    plt.close()


    print(

        f"\nTraining curves saved to: "

        f"{output_path}"

    )


# ============================================================
# MAIN
# ============================================================

def main():


    args = parse_args()


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(

        args.seed

    )


    np.random.seed(

        args.seed

    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(

            args.seed

        )


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(

        "\nLoading multi-label dataset..."

    )


    (

        train_dataset,

        val_dataset,

        test_dataset,

        class_names

    ) = build_datasets(

        data_dir=args.data_dir,

        val_size=0.15,

        test_size=0.15,

        seed=args.seed

    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )


    num_classes = len(

        class_names

    )


    print(

        f"\nTraining images: "

        f"{len(train_dataset)}"

    )


    print(

        f"Validation images: "

        f"{len(val_dataset)}"

    )


    print(

        f"Testing images: "

        f"{len(test_dataset)}"

    )


    print(

        f"\nNumber of classes: "

        f"{num_classes}"

    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_model(

        num_classes=num_classes,

        device=device,

        pretrained=True

    )


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.BCEWithLogitsLoss()


    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    os.makedirs(

        args.checkpoint_dir,

        exist_ok=True

    )


    checkpoint_path = os.path.join(

        args.checkpoint_dir,

        "best_multi_label_model.pt"

    )


    # ========================================================
    # HISTORY
    # ========================================================

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": []

    }


    # ========================================================
    # PHASE 1
    # FROZEN BACKBONE
    # ========================================================

    print(

        "\nStarting Phase 1..."

    )


    model.freeze_backbone()


    optimizer = torch.optim.AdamW(

        filter(

            lambda p: p.requires_grad,

            model.parameters()

        ),

        lr=args.phase1_lr,

        weight_decay=args.weight_decay

    )


    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=2

        )

    )


    train_phase(

        phase_name="Phase 1 (Frozen Backbone)",

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        num_epochs=args.phase1_epochs,

        checkpoint_path=checkpoint_path,

        history=history,

        class_names=class_names,

        threshold=args.threshold

    )


    # ========================================================
    # PHASE 2
    # FINE TUNE LAYER4
    # ========================================================

    print(

        "\nStarting Phase 2..."

    )


    model.unfreeze_last_blocks(

        blocks=("layer4",)

    )


    optimizer = torch.optim.AdamW(

        filter(

            lambda p: p.requires_grad,

            model.parameters()

        ),

        lr=args.phase2_lr,

        weight_decay=args.weight_decay

    )


    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=2

        )

    )


    train_phase(

        phase_name="Phase 2 (Fine Tuning)",

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        num_epochs=args.phase2_epochs,

        checkpoint_path=checkpoint_path,

        history=history,

        class_names=class_names,

        threshold=args.threshold

    )


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    plot_history(

        history,

        os.path.join(

            args.checkpoint_dir,

            "multi_label_training_curves.png"

        )

    )


    print(

        "\nTraining completed!"

    )


    print(

        f"\nBest model saved at:\n"

        f"{checkpoint_path}"

    )


if __name__ == "__main__":

    main()