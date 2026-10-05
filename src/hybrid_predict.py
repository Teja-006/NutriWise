"""
hybrid_predict.py
-----------------

Hybrid Indian Food Prediction Pipeline

Pipeline:

Input Image
    ↓
Segmentation Model
    ↓
Food Mask
    ↓
Background Removal
    ↓
Segmented Food Image
    ↓
Multi-Label Classifier
    ↓
Single-Label Classifier
    ↓
Final Results


Usage:

python src/hybrid_predict.py --image data/test8.jpg
"""


import argparse
import os

import numpy as np

import torch

from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder


# ============================================================
# IMPORT MODELS
# ============================================================

from segmentation_model import (
    build_segmentation_model,
    get_device
)

from model import (
    MultiLabelResNet50
)

from model1 import (
    ResNet50FoodClassifier
)


# ============================================================
# MULTI-LABEL CLASS NAMES
# ============================================================

MULTI_CLASS_NAMES = [

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
# GET SINGLE-LABEL CLASS NAMES
# ============================================================

def get_single_class_names(data_dir):

    dataset = ImageFolder(data_dir)

    return dataset.classes


# ============================================================
# SEGMENTATION TRANSFORM
# ============================================================

def get_segmentation_transform():

    return transforms.Compose([

        transforms.Resize(

            (512, 512)

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
# CLASSIFICATION TRANSFORM
# ============================================================

def get_classification_transform():

    return transforms.Compose([

        transforms.Resize(

            (224, 224)

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
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model(

    checkpoint_path,

    device

):

    print(

        "\nLoading segmentation model..."

    )


    if not os.path.exists(

        checkpoint_path

    ):

        raise FileNotFoundError(

            f"Segmentation checkpoint not found:\n"

            f"{checkpoint_path}"

        )


    model = build_segmentation_model(

        num_classes=51,

        device=device,

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    if (

        "model_state_dict"

        in checkpoint

    ):

        state_dict = (

            checkpoint["model_state_dict"]

        )

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # strict=False
    #
    # Your checkpoint contains aux_classifier weights.
    # The current inference model does not use them.
    # --------------------------------------------------------

    model.load_state_dict(

        state_dict,

        strict=False

    )


    model.eval()


    print(

        "Segmentation model loaded successfully!"

    )


    return model


# ============================================================
# LOAD MULTI-LABEL MODEL
# ============================================================

def load_multi_label_model(

    checkpoint_path,

    device

):

    print(

        "\nLoading multi-label model..."

    )


    if not os.path.exists(

        checkpoint_path

    ):

        raise FileNotFoundError(

            f"Multi-label checkpoint not found:\n"

            f"{checkpoint_path}"

        )


    model = MultiLabelResNet50(

        num_classes=len(

            MULTI_CLASS_NAMES

        ),

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    if (

        "model_state_dict"

        in checkpoint

    ):

        state_dict = (

            checkpoint["model_state_dict"]

        )

    else:

        state_dict = checkpoint


    model.load_state_dict(

        state_dict

    )


    model.to(device)

    model.eval()


    print(

        "Multi-label model loaded successfully!"

    )


    return model


# ============================================================
# LOAD SINGLE-LABEL MODEL
# ============================================================

def load_single_label_model(

    checkpoint_path,

    num_classes,

    device

):

    print(

        "\nLoading single-label model..."

    )


    if not os.path.exists(

        checkpoint_path

    ):

        raise FileNotFoundError(

            f"Single-label checkpoint not found:\n"

            f"{checkpoint_path}"

        )


    model = ResNet50FoodClassifier(

        num_classes=num_classes,

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    if (

        "model_state_dict"

        in checkpoint

    ):

        state_dict = (

            checkpoint["model_state_dict"]

        )

    else:

        state_dict = checkpoint


    model.load_state_dict(

        state_dict

    )


    model.to(device)

    model.eval()


    print(

        "Single-label model loaded successfully!"

    )


    return model


# ============================================================
# SEGMENT IMAGE
# ============================================================

def segment_image(

    image_path,

    model,

    device

):

    print(

        "\nSegmenting image..."

    )


    if not os.path.exists(

        image_path

    ):

        raise FileNotFoundError(

            f"Image not found:\n"

            f"{image_path}"

        )


    original_image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    original_width, original_height = (

        original_image.size

    )


    transform = (

        get_segmentation_transform()

    )


    input_tensor = (

        transform(

            original_image

        )

        .unsqueeze(0)

        .to(device)

    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(

            input_tensor

        )


        prediction = torch.argmax(

            output,

            dim=1

        )


    prediction = (

        prediction

        .squeeze(0)

        .cpu()

        .numpy()

    )


    # ========================================================
    # FOOD MASK
    #
    # Class 0 = Background
    # Class > 0 = Food
    # ========================================================

    food_mask = (

        prediction != 0

    )


    # ========================================================
    # RESIZE MASK TO ORIGINAL IMAGE SIZE
    # ========================================================

    mask_image = Image.fromarray(

        (

            food_mask.astype(np.uint8)

            * 255

        )

    )


    mask_image = mask_image.resize(

        (

            original_width,

            original_height

        ),

        Image.Resampling.NEAREST

    )


    mask_array = (

        np.array(mask_image)

        > 0

    )


    # ========================================================
    # ORIGINAL IMAGE ARRAY
    # ========================================================

    original_array = np.array(

        original_image

    )


    # ========================================================
    # REMOVE BACKGROUND
    # ========================================================

    segmented_array = (

        original_array.copy()

    )


    segmented_array[

        ~mask_array

    ] = [

        0,

        0,

        0

    ]


    segmented_image = Image.fromarray(

        segmented_array

    )


    return (

        original_image,

        mask_image,

        segmented_image,

        prediction

    )


# ============================================================
# SAVE SEGMENTATION OUTPUT
# ============================================================

def save_segmentation_output(

    mask_image,

    segmented_image,

    output_dir

):

    os.makedirs(

        output_dir,

        exist_ok=True

    )


    mask_path = os.path.join(

        output_dir,

        "food_mask.png"

    )


    segmented_path = os.path.join(

        output_dir,

        "segmented_food.png"

    )


    mask_image.save(

        mask_path

    )


    segmented_image.save(

        segmented_path

    )


    print(

        "\nFood mask saved to:"

    )

    print(

        mask_path

    )


    print(

        "\nSegmented food image saved to:"

    )

    print(

        segmented_path

    )


# ============================================================
# MULTI-LABEL PREDICTION
# ============================================================

def predict_multi_label(

    image,

    model,

    device,

    threshold

):

    transform = (

        get_classification_transform()

    )


    input_tensor = (

        transform(

            image

        )

        .unsqueeze(0)

        .to(device)

    )


    with torch.no_grad():

        outputs = model(

            input_tensor

        )


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


    for index, probability in enumerate(

        probabilities

    ):


        if probability >= threshold:


            predictions.append({

                "class":

                    MULTI_CLASS_NAMES[index],

                "probability":

                    float(probability)

            })


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    predictions = sorted(

        predictions,

        key=lambda x:

            x["probability"],

        reverse=True

    )


    return predictions


# ============================================================
# SINGLE-LABEL PREDICTION
# ============================================================

def predict_single_label(

    image,

    model,

    class_names,

    device,

    top_k

):

    transform = (

        get_classification_transform()

    )


    input_tensor = (

        transform(

            image

        )

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

        .squeeze(0)

        .cpu()

        .numpy()

    )


    indices = (

        indices

        .squeeze(0)

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


# ============================================================
# NORMALIZE FOOD NAME
#
# Helps compare names like:
#
# plain-rice
# Plain Rice
# plain_rice
# ============================================================

def normalize_food_name(

    name

):

    name = (

        name.lower()

    )


    name = (

        name.replace(

            "-",

            " "

        )

    )


    name = (

        name.replace(

            "_",

            " "

        )

    )


    name = (

        name.replace(

            "&",

            "and"

        )

    )


    return (

        " ".join(

            name.split()

        )

    )


# ============================================================
# CHECK SINGLE LABEL AGAINST MULTI LABEL
# ============================================================

def verify_predictions(

    multi_predictions,

    single_predictions

):

    multi_names = []


    for prediction in (

        multi_predictions

    ):


        multi_names.append(

            normalize_food_name(

                prediction["class"]

            )

        )


    verification_results = []


    for prediction in (

        single_predictions

    ):


        single_name = (

            normalize_food_name(

                prediction["class"]

            )

        )


        found = False


        for multi_name in (

            multi_names

        ):


            # ------------------------------------------------
            # Approximate matching
            # ------------------------------------------------

            if (

                single_name == multi_name

                or

                single_name in multi_name

                or

                multi_name in single_name

            ):


                found = True

                break


        verification_results.append({

            "class":

                prediction["class"],

            "probability":

                prediction["probability"],

            "found_in_multi":

                found

        })


    return verification_results


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_results(

    multi_predictions,

    single_predictions,

    verification_results

):


    print(

        "\n"

        "============================================================"

    )


    print(

        "MULTI-LABEL FOOD DETECTION"

    )


    print(

        "============================================================\n"

    )


    if len(

        multi_predictions

    ) == 0:


        print(

            "No foods detected above threshold."

        )


    else:


        for i, prediction in enumerate(

            multi_predictions,

            start=1

        ):


            print(

                f"{i}. "

                f"{prediction['class']} "

                f"- "

                f"{prediction['probability'] * 100:.2f}%"

            )


    print(

        "\n"

        "============================================================"

    )


    print(

        "SINGLE-LABEL VERIFICATION"

    )


    print(

        "============================================================\n"

    )


    for i, prediction in enumerate(

        single_predictions,

        start=1

    ):


        print(

            f"{i}. "

            f"{prediction['class']} "

            f"- "

            f"{prediction['probability'] * 100:.2f}%"

        )


    print(

        "\n"

        "============================================================"

    )


    print(

        "HYBRID VERIFICATION"

    )


    print(

        "============================================================\n"

    )


    for result in (

        verification_results

    ):


        if result[

            "found_in_multi"

        ]:


            status = (

                "✓ CONFIRMED BY MULTI-LABEL"

            )


        else:


            status = (

                "⚠ NOT IN MULTI-LABEL RESULTS"

            )


        print(

            f"{result['class']} "

            f"({result['probability'] * 100:.2f}%) "

            f"→ {status}"

        )


# ============================================================
# MAIN
# ============================================================

def main():


    parser = argparse.ArgumentParser(

        description=

        "NutriWise Hybrid Food Prediction"

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
    # SEGMENTATION CHECKPOINT
    # --------------------------------------------------------

    parser.add_argument(

        "--seg_checkpoint",

        type=str,

        default=

        "checkpoints/best_segmentation_model.pt"

    )


    # --------------------------------------------------------
    # MULTI-LABEL CHECKPOINT
    # --------------------------------------------------------

    parser.add_argument(

        "--multi_checkpoint",

        type=str,

        default=

        "checkpoints/best_multi_label_model.pt"

    )


    # --------------------------------------------------------
    # SINGLE-LABEL CHECKPOINT
    # --------------------------------------------------------

    parser.add_argument(

        "--single_checkpoint",

        type=str,

        default=

        "checkpoints/best_single_label_model.pt"

    )


    # --------------------------------------------------------
    # SINGLE LABEL DATA
    # --------------------------------------------------------

    parser.add_argument(

        "--single_data_dir",

        type=str,

        default=

        "data/Indian Food"

    )


    # --------------------------------------------------------
    # MULTI-LABEL THRESHOLD
    # --------------------------------------------------------

    parser.add_argument(

        "--threshold",

        type=float,

        default=0.50

    )


    # --------------------------------------------------------
    # SINGLE LABEL TOP K
    # --------------------------------------------------------

    parser.add_argument(

        "--top_k",

        type=int,

        default=5

    )


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(

        "--output_dir",

        type=str,

        default=

        "hybrid_output"

    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # SINGLE CLASS NAMES
    # ========================================================

    print(

        "\nLoading single-label class names..."

    )


    single_class_names = (

        get_single_class_names(

            args.single_data_dir

        )

    )


    print(

        f"Single-label classes: "

        f"{len(single_class_names)}"

    )


    print(

        f"Multi-label classes: "

        f"{len(MULTI_CLASS_NAMES)}"

    )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    segmentation_model = (

        load_segmentation_model(

            checkpoint_path=

                args.seg_checkpoint,

            device=

                device

        )

    )


    multi_model = (

        load_multi_label_model(

            checkpoint_path=

                args.multi_checkpoint,

            device=

                device

        )

    )


    single_model = (

        load_single_label_model(

            checkpoint_path=

                args.single_checkpoint,

            num_classes=

                len(

                    single_class_names

                ),

            device=

                device

        )

    )


    # ========================================================
    # SEGMENT IMAGE
    # ========================================================

    (

        original_image,

        food_mask,

        segmented_image,

        segmentation_prediction

    ) = segment_image(

        image_path=

            args.image,

        model=

            segmentation_model,

        device=

            device

    )


    # ========================================================
    # SAVE SEGMENTATION
    # ========================================================

    save_segmentation_output(

        mask_image=

            food_mask,

        segmented_image=

            segmented_image,

        output_dir=

            args.output_dir

    )


    # ========================================================
    # MULTI-LABEL PREDICTION
    # ========================================================

    print(

        "\nRunning multi-label prediction..."

    )


    multi_predictions = (

        predict_multi_label(

            image=

                segmented_image,

            model=

                multi_model,

            device=

                device,

            threshold=

                args.threshold

        )

    )


    # ========================================================
    # SINGLE-LABEL PREDICTION
    # ========================================================

    print(

        "\nRunning single-label verification..."

    )


    single_predictions = (

        predict_single_label(

            image=

                segmented_image,

            model=

                single_model,

            class_names=

                single_class_names,

            device=

                device,

            top_k=

                args.top_k

        )

    )


    # ========================================================
    # VERIFY
    # ========================================================

    verification_results = (

        verify_predictions(

            multi_predictions=

                multi_predictions,

            single_predictions=

                single_predictions

        )

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_results(

        multi_predictions=

            multi_predictions,

        single_predictions=

            single_predictions,

        verification_results=

            verification_results

    )


    print(

        "\n"

        "============================================================"

    )


    print(

        "HYBRID PREDICTION COMPLETE"

    )


    print(

        "============================================================\n"

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()