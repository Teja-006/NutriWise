"""
predict.py
----------
Run inference on a single image using the trained
single-label ResNet50 model.

Usage:

python src/predict.py --image path/to/image.jpg
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model1 import ResNet50FoodClassifier, get_device


def get_class_names(data_dir):

    dataset = ImageFolder(data_dir)

    return dataset.classes


def load_model_for_inference(
    checkpoint_path,
    num_classes,        
    device
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )


    model = ResNet50FoodClassifier(

        num_classes=num_classes,

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint["model_state_dict"]

    )


    model.to(device)

    model.eval()


    return model


def predict_image(
    image_path,
    model,
    class_names,
    device,
    image_size=224,
    top_k=5
):

    if not os.path.exists(image_path):

        raise FileNotFoundError(

            f"Image not found: {image_path}"

        )


    image = Image.open(

        image_path

    ).convert("RGB")


    transform = transforms.Compose([

        transforms.Resize(

            (image_size, image_size)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )

    ])


    input_tensor = (

        transform(image)

        .unsqueeze(0)

        .to(device)

    )


    with torch.no_grad():

        outputs = model(

            input_tensor

        )


        probabilities = torch.softmax(

            outputs,

            dim=1

        )


    probabilities, indices = torch.topk(

        probabilities,

        top_k

    )


    probabilities = (

        probabilities

        .squeeze()

        .cpu()

        .numpy()

    )


    indices = (

        indices

        .squeeze()

        .cpu()

        .numpy()

    )


    predictions = []


    for probability, index in zip(

        probabilities,

        indices

    ):

        predictions.append({

            "class":

                class_names[index],

            "probability":

                float(probability)

        })


    return predictions


def main():

    parser = argparse.ArgumentParser(

        description=

        "Predict Indian food from an image"

    )


    parser.add_argument(

        "--image",

        type=str,

        required=True

    )


    parser.add_argument(

        "--checkpoint",

        type=str,

        default=

        "checkpoints/best_single_label_model.pt"

    )


    parser.add_argument(

        "--data_dir",

        type=str,

        default="data/Indian Food"

    )


    parser.add_argument(

        "--top_k",

        type=int,

        default=5

    )


    args = parser.parse_args()


    device = get_device()


    # -------------------------
    # Load class names
    # -------------------------

    class_names = get_class_names(

        args.data_dir

    )


    print(

        f"\nLoaded {len(class_names)} classes."

    )


    # -------------------------
    # Load trained model
    # -------------------------

    model = load_model_for_inference(

        checkpoint_path=args.checkpoint,

        num_classes=len(class_names),

        device=device

    )


    # -------------------------
    # Predict
    # -------------------------

    predictions = predict_image(

        image_path=args.image,

        model=model,

        class_names=class_names,

        device=device,

        top_k=args.top_k

    )


    print(

        "\n============================"

    )

    print(

        "TOP PREDICTIONS"

    )

    print(

        "============================\n"

    )


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


if __name__ == "__main__":

    main()