"""
predict.py
----------
Run inference on a single image using the trained
multi-label ResNet50 food classifier.

Usage:
    python src/predict.py --image data/test6.jpeg
"""

import argparse
import os

import torch

from PIL import Image
from torchvision import transforms

from model import MultiLabelResNet50, get_device


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [

    "Aloo Dry fry",
    "Avakaya Muddha Papu Rice",
    "Baby-Corn & Capsicum-Dry",
    "Cabbage Pakodi",
    "Cabbage fry",
    "Capsicum Paneer Curry",
    "Chakar-Pongal",
    "Chole-Masala",
    "Cluster Beans Curry",
    "Cucumber-Raitha",
    "Gobi Masala Curry",
    "Gutti Vankaya Curry",
    "Jeera Rice",
    "Mixed Curry",
    "Muskmelon",
    "Rajma",
    "Rasgulla",
    "Sambar",
    "Tomato Rasam",
    "Vankaya-Ali-Karam",
    "Veg-Biriyani",
    "aloo-curry",
    "curd",
    "dal",
    "fresh-chutney",
    "green-salad",
    "Moong-Beans-Curry",
    "khichdi",
    "lemon-rice",
    "live-roti-with-ghee",
    "non-spicy-curry-bottle-gourd",
    "papad",
    "plain-rice",
    "watermelon",
    "Aloo-Fry",
    "Banana",
    "Mix-Fruit",
    "Non-Spicy-Baby-Corn & Capsicum-Dry",
    "Sweet",
    "Tomato-Rice",
    "fried-papad-rings",
    "gravy",
    "ivy-gourd-fry",
    "mango-pickle",
    "papad-chat",
    "pepper-rasam",
    "pineapple",
    "corn-fry",
    "paneer-curry",
    "semiya"

]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_for_inference(
    checkpoint_path,
    num_classes,
    device
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )


    # Create MULTI-LABEL model
    model = MultiLabelResNet50(

        num_classes=num_classes,

        pretrained=False

    )


    # Load checkpoint
    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    # Load trained weights
    model.load_state_dict(

        checkpoint["model_state_dict"]

    )


    model.to(device)

    model.eval()


    return model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

def get_transform(image_size=224):

    return transforms.Compose([

        transforms.Resize(

            (image_size, image_size)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ])


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(

    image_path,

    model,

    class_names,

    device,

    threshold=0.5,

    image_size=224

):


    if not os.path.exists(image_path):

        raise FileNotFoundError(

            f"Image not found: {image_path}"

        )


    # Load image
    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # Transform image
    transform = get_transform(

        image_size

    )


    input_tensor = (

        transform(image)

        .unsqueeze(0)

        .to(device)

    )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.no_grad():

        outputs = model(

            input_tensor

        )


        # Multi-label classification uses SIGMOID
        probabilities = torch.sigmoid(

            outputs

        )


    probabilities = (

        probabilities

        .squeeze(0)

        .cpu()

        .numpy()

    )


    predictions = []


    # ========================================================
    # GET ALL FOODS ABOVE THRESHOLD
    # ========================================================

    for index, probability in enumerate(

        probabilities

    ):


        if probability >= threshold:


            predictions.append({

                "class":

                    class_names[index],

                "probability":

                    float(probability)

            })


    # ========================================================
    # SORT BY CONFIDENCE
    # ========================================================

    predictions = sorted(

        predictions,

        key=lambda x: x["probability"],

        reverse=True

    )


    return predictions, probabilities


# ============================================================
# MAIN
# ============================================================

def main():


    parser = argparse.ArgumentParser(

        description="Multi-label Indian Food Prediction"

    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    parser.add_argument(

        "--image",

        type=str,

        required=True

    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    parser.add_argument(

        "--checkpoint",

        type=str,

        default="checkpoints/best_multi_label_model.pt"

    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    parser.add_argument(

        "--threshold",

        type=float,

        default=0.5

    )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    parser.add_argument(

        "--top_k",

        type=int,

        default=10

    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # CLASS NAMES
    # ========================================================

    class_names = CLASS_NAMES


    print(

        f"\nLoaded {len(class_names)} classes."

    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model_for_inference(

        checkpoint_path=args.checkpoint,

        num_classes=len(class_names),

        device=device

    )


    print(

        "Model loaded successfully!"

    )


    # ========================================================
    # PREDICT
    # ========================================================

    predictions, probabilities = predict_image(

        image_path=args.image,

        model=model,

        class_names=class_names,

        device=device,

        threshold=args.threshold

    )


    # ========================================================
    # PRINT DETECTED FOODS
    # ========================================================

    print(

        "\n============================"

    )

    print(

        "DETECTED FOODS"

    )

    print(

        "============================\n"

    )


    if len(predictions) == 0:


        print(

            f"No foods detected above threshold "
            f"{args.threshold}"

        )


        print(

            "\nTry a lower threshold, for example:"

        )


        print(

            "python src/predict.py "
            "--image data/test6.jpeg "
            "--threshold 0.3"

        )


    else:


        for i, prediction in enumerate(

            predictions,

            start=1

        ):


            print(

                f"{i}. "

                f"{prediction['class']} "

                f"- "

                f"{prediction['probability'] * 100:.2f}%"

            )


    # ========================================================
    # SHOW TOP K PREDICTIONS
    # ========================================================

    print(

        "\n============================"

    )

    print(

        f"TOP {args.top_k} PREDICTIONS"

    )

    print(

        "============================\n"

    )


    top_indices = (

        probabilities.argsort()[::-1]

        [:args.top_k]

    )


    for rank, index in enumerate(

        top_indices,

        start=1

    ):


        print(

            f"{rank}. "

            f"{class_names[index]} "

            f"- "

            f"{probabilities[index] * 100:.2f}%"

        )


if __name__ == "__main__":

    main()