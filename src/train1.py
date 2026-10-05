"""
train.py
--------
Two-phase transfer learning for multi-label Indian food classification.

Phase 1: backbone frozen, train only the classification head.
Phase 2: unfreeze later ResNet blocks (e.g. layer4), fine-tune at a lower LR.

Usage (from the NutriWise/ root):
    python src/train.py --data_dir data --dataset_format csv \
        --phase1_epochs 5 --phase2_epochs 15 --batch_size 32

Run src/inspect_dataset.py first to determine --dataset_format.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model1 import build_model, get_device


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train ResNet50 for Indian Food Classification"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/Indian Food"
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
        "--image_size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=5,
        help="Train classification head only"
    )

    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=15,
        help="Fine-tune ResNet layer4"
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
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None
):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if is_train:

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


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
    history
):

    print("\n" + "=" * 60)

    print(phase_name)

    print(
        f"Trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )

    print("=" * 60)

    best_accuracy = 0.0


    for epoch in range(
        1,
        num_epochs + 1
    ):

        start_time = time.time()


        # ------------------
        # Training
        # ------------------

        train_loss, train_accuracy = run_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            device=device,

            optimizer=optimizer

        )


        # ------------------
        # Validation
        # ------------------

        val_loss, val_accuracy = run_epoch(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device

        )


        scheduler.step(
            val_loss
        )


        elapsed = (
            time.time()
            - start_time
        )


        print(

            f"[{phase_name}] "

            f"Epoch {epoch}/{num_epochs} "

            f"({elapsed:.1f}s) | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Acc: {train_accuracy:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_accuracy:.4f}"

        )


        # Save history

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )


        # ------------------
        # Save Best Model
        # ------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy


            torch.save(

                {

                    "model_state_dict":

                        model.state_dict(),

                    "optimizer_state_dict":

                        optimizer.state_dict(),

                    "epoch":

                        epoch,

                    "val_accuracy":

                        val_accuracy

                },

                checkpoint_path

            )


            print(

                f"  -> New best model! "

                f"Validation Accuracy: "

                f"{val_accuracy:.4f}"

            )


    return best_accuracy


def plot_history(
    history,
    output_path
):

    plt.figure(
        figsize=(12, 5)
    )


    # ------------------
    # Loss Graph
    # ------------------

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


    # ------------------
    # Accuracy Graph
    # ------------------

    plt.subplot(
        1,
        2,
        2
    )


    plt.plot(

        history["train_accuracy"],

        label="Training Accuracy"

    )


    plt.plot(

        history["val_accuracy"],

        label="Validation Accuracy"

    )


    plt.title(
        "Training and Validation Accuracy"
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


    print(

        f"\nTraining curves saved to: "

        f"{output_path}"

    )


def main():

    args = parse_args()


    # ------------------
    # Reproducibility
    # ------------------

    torch.manual_seed(
        args.seed
    )


    np.random.seed(
        args.seed
    )


    # ------------------
    # Device
    # ------------------

    device = get_device()


    # ------------------
    # Dataset
    # ------------------

    print(
        "\nLoading dataset..."
    )


    train_loader, val_loader, class_names = (

        get_dataloaders(

            data_dir=args.data_dir,

            image_size=args.image_size,

            batch_size=args.batch_size

        )

    )


    num_classes = len(
        class_names
    )


    print(
        f"\nNumber of classes: "
        f"{num_classes}"
    )


    # ------------------
    # Model
    # ------------------

    model = build_model(

        num_classes=num_classes,

        device=device,

        pretrained=True

    )


    # ------------------
    # Loss Function
    # ------------------

    criterion = nn.CrossEntropyLoss()


    # ------------------
    # Checkpoint Folder
    # ------------------

    os.makedirs(

        args.checkpoint_dir,

        exist_ok=True

    )


    checkpoint_path = os.path.join(

        args.checkpoint_dir,

        "best_single_label_model.pt"

    )


    # ------------------
    # Training History
    # ------------------

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": []

    }


    # ==================================
    # PHASE 1
    # Train Classification Head
    # ==================================

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

        torch.optim.lr_scheduler.

        ReduceLROnPlateau(

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

        history=history

    )


    # ==================================
    # PHASE 2
    # Fine Tune Layer 4
    # ==================================

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

        torch.optim.lr_scheduler.

        ReduceLROnPlateau(

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

        history=history

    )


    # ------------------
    # Save Graph
    # ------------------

    plot_history(

        history,

        os.path.join(

            args.checkpoint_dir,

            "single_label_training_curves.png"

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