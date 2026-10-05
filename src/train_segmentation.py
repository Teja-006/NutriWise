"""
train_segmentation.py
---------------------
Train DeepLabV3-ResNet50 for Indian food segmentation.

Run from the NutriWise root:

python src/train_segmentation.py
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from segmentation_dataset import build_segmentation_datasets
from segmentation_model import (
    build_segmentation_model,
    get_device
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description="Train Indian Food Segmentation Model"

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

        "--epochs",

        type=int,

        default=20

    )


    parser.add_argument(

        "--batch_size",

        type=int,

        default=4

    )


    parser.add_argument(

        "--learning_rate",

        type=float,

        default=1e-4

    )


    parser.add_argument(

        "--image_size",

        type=int,

        default=512

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


# ============================================================
# PIXEL ACCURACY
# ============================================================

def calculate_pixel_accuracy(

    predictions,

    masks

):

    predictions = torch.argmax(

        predictions,

        dim=1

    )


    correct_pixels = (

        predictions == masks

    ).sum().item()


    total_pixels = masks.numel()


    accuracy = (

        correct_pixels /

        total_pixels

    )


    return accuracy


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(

    model,

    loader,

    criterion,

    device,

    optimizer=None

):


    is_train = (

        optimizer is not None

    )


    if is_train:

        model.train()


    else:

        model.eval()


    total_loss = 0.0

    total_correct = 0

    total_pixels = 0


    # --------------------------------------------------------
    # GRADIENT MODE
    # --------------------------------------------------------

    with torch.set_grad_enabled(

        is_train

    ):


        for images, masks in loader:


            images = images.to(

                device,

                non_blocking=True

            )


            masks = masks.to(

                device,

                non_blocking=True

            ).long()


            # ------------------------------------------------
            # TRAINING
            # ------------------------------------------------

            if is_train:

                optimizer.zero_grad()


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            outputs = model(

                images

            )


            loss = criterion(

                outputs,

                masks

            )


            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            if is_train:


                loss.backward()


                optimizer.step()


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            total_loss += (

                loss.item()

                * images.size(0)

            )


            # ------------------------------------------------
            # PIXEL ACCURACY
            # ------------------------------------------------

            predictions = torch.argmax(

                outputs,

                dim=1

            )


            total_correct += (

                predictions == masks

            ).sum().item()


            total_pixels += (

                masks.numel()

            )


    average_loss = (

        total_loss /

        len(loader.dataset)

    )


    pixel_accuracy = (

        total_correct /

        total_pixels

    )


    return (

        average_loss,

        pixel_accuracy

    )


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


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

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

        "Segmentation Loss"

    )


    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Loss"

    )


    plt.legend()


    # --------------------------------------------------------
    # PIXEL ACCURACY
    # --------------------------------------------------------

    plt.subplot(

        1,

        2,

        2

    )


    plt.plot(

        history["train_accuracy"],

        label="Training Pixel Accuracy"

    )


    plt.plot(

        history["val_accuracy"],

        label="Validation Pixel Accuracy"

    )


    plt.title(

        "Segmentation Pixel Accuracy"

    )


    plt.xlabel(

        "Epoch"

    )


    plt.ylabel(

        "Pixel Accuracy"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        output_path

    )


    print(

        f"\nTraining graph saved to: "

        f"{output_path}"

    )


# ============================================================
# MAIN
# ============================================================

def main():


    args = parse_args()


    # --------------------------------------------------------
    # RANDOM SEED
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(

        "\nLoading segmentation dataset..."

    )


    train_dataset, val_dataset = (

        build_segmentation_datasets(

            data_dir=args.data_dir,

            image_size=args.image_size,

            seed=args.seed

        )

    )


    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True

    )


    val_loader = torch.utils.data.DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )


    print(

        f"\nTraining samples: "

        f"{len(train_dataset)}"

    )


    print(

        f"Validation samples: "

        f"{len(val_dataset)}"

    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    num_classes = 51


    print(

        f"\nNumber of segmentation classes: "

        f"{num_classes}"

    )


    model = build_segmentation_model(

        num_classes=num_classes,

        device=device,

        pretrained=True

    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.learning_rate,

        weight_decay=args.weight_decay

    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=2

        )

    )


    # --------------------------------------------------------
    # CHECKPOINT FOLDER
    # --------------------------------------------------------

    os.makedirs(

        args.checkpoint_dir,

        exist_ok=True

    )


    checkpoint_path = os.path.join(

        args.checkpoint_dir,

        "best_segmentation_model.pt"

    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": []

    }


    best_val_accuracy = 0.0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print(

        "\nStarting segmentation training..."

    )


    print(

        "=" * 60

    )


    for epoch in range(

        1,

        args.epochs + 1

    ):


        start_time = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (

            run_epoch(

                model=model,

                loader=train_loader,

                criterion=criterion,

                device=device,

                optimizer=optimizer

            )

        )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_loss, val_accuracy = (

            run_epoch(

                model=model,

                loader=val_loader,

                criterion=criterion,

                device=device

            )

        )


        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(

            val_loss

        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        elapsed = (

            time.time()

            - start_time

        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(

            f"\nEpoch "

            f"{epoch}/{args.epochs} "

            f"({elapsed:.1f}s)"

        )


        print(

            f"Train Loss: "

            f"{train_loss:.4f}"

        )


        print(

            f"Train Pixel Accuracy: "

            f"{train_accuracy:.4f}"

        )


        print(

            f"Val Loss: "

            f"{val_loss:.4f}"

        )


        print(

            f"Val Pixel Accuracy: "

            f"{val_accuracy:.4f}"

        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:


            best_val_accuracy = (

                val_accuracy

            )


            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict":

                        model.state_dict(),

                    "optimizer_state_dict":

                        optimizer.state_dict(),

                    "val_pixel_accuracy":

                        val_accuracy,

                    "val_loss":

                        val_loss,

                    "num_classes":

                        num_classes

                },

                checkpoint_path

            )


            print(

                "\n🔥 New best segmentation model!"

            )


            print(

                f"Validation Pixel Accuracy: "

                f"{val_accuracy:.4f}"

            )


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    graph_path = os.path.join(

        args.checkpoint_dir,

        "segmentation_training_curves.png"

    )


    plot_history(

        history,

        graph_path

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "SEGMENTATION TRAINING COMPLETED!"

    )


    print(

        "=" * 60

    )


    print(

        f"\nBest Validation Pixel Accuracy: "

        f"{best_val_accuracy:.4f}"

    )


    print(

        f"\nBest model saved at:\n"

        f"{checkpoint_path}"

    )


if __name__ == "__main__":

    main()