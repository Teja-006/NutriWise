"""
predict_hybrid.py
-----------------

Hybrid Food Prediction System.

Uses:

1. Multi-label model
   - Detects multiple foods in one image.

2. Single-label model
   - Predicts the most likely food from
     the Indian Food dataset.

Usage:

python src/predict_hybrid.py --image data/test6.jpeg
"""


import argparse
import os

import torch

from PIL import Image

from torchvision import transforms


# ============================================================
# IMPORT MODELS
# ============================================================

from model import MultiLabelResNet50

from model1 import ResNet50FoodClassifier, get_device


# ============================================================
# MULTI-LABEL CLASS NAMES
# ============================================================

MULTI_LABEL_CLASSES = [

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
# SINGLE-LABEL CLASS NAMES
# ============================================================

SINGLE_LABEL_CLASSES = [

    "adhirasam",
    "aloo_gobi",
    "aloo_matar",
    "aloo_methi",
    "aloo_shimla_mirch",
    "aloo_tikki",
    "anarsa",
    "ariselu",
    "bandar_laddu",
    "basundi",
    "bhatura",
    "bhindi_masala",
    "biryani",
    "boondi",
    "butter_chicken",
    "chak_hao_kheer",
    "cham_cham",
    "chana_masala",
    "chapati",
    "chhena_kheeri",
    "chicken_razala",
    "chicken_tikka",
    "chicken_tikka_masala",
    "chikki",
    "daal_baati_churma",
    "daal_puri",
    "dal_makhani",
    "dal_tadka",
    "dharwad_pedha",
    "doodhpak",
    "double_ka_meetha",
    "dum_aloo",
    "gajar_ka_halwa",
    "gavvalu",
    "ghevar",
    "gulab_jamun",
    "imarti",
    "jalebi",
    "kachori",
    "kadai_paneer",
    "kadhi_pakoda",
    "kajjikaya",
    "kakinada_khaja",
    "kalakand",
    "karela_bharta",
    "kofta",
    "kuzhi_paniyaram",
    "lassi",
    "ledikeni",
    "litti_chokha",
    "lyangcha",
    "maach_jhol",
    "makki_di_roti_sarson_da_saag",
    "malapua",
    "misi_roti",
    "misti_doi",
    "modak",
    "mysore_pak",
    "naan",
    "navrattan_korma",
    "palak_paneer",
    "paneer_butter_masala",
    "phirni",
    "pithe",
    "poha",
    "poornalu",
    "pootharekulu",
    "qubani_ka_meetha",
    "rabri",
    "rasgulla",
    "ras_malai",
    "sandesh",
    "shankarpali",
    "sheera",
    "sheer_korma",
    "shrikhand",
    "sohan_halwa",
    "sohan_papdi",
    "sutar_feni",
    "unni_appam"

]


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
# LOAD MULTI-LABEL MODEL
# ============================================================

def load_multi_label_model(

    checkpoint_path,
    device

):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(

            f"Multi-label checkpoint not found: "
            f"{checkpoint_path}"

        )


    model = MultiLabelResNet50(

        num_classes=len(
            MULTI_LABEL_CLASSES
        ),

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model.to(device)

    model.eval()


    return model


# ============================================================
# LOAD SINGLE-LABEL MODEL
# ============================================================

def load_single_label_model(

    checkpoint_path,
    device

):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(

            f"Single-label checkpoint not found: "
            f"{checkpoint_path}"

        )


    model = ResNet50FoodClassifier(

        num_classes=len(
            SINGLE_LABEL_CLASSES
        ),

        pretrained=False

    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model.to(device)

    model.eval()


    return model


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(

    image_path,
    device

):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(

            f"Image not found: "
            f"{image_path}"

        )


    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    transform = get_transform()


    input_tensor = (

        transform(image)

        .unsqueeze(0)

        .to(device)

    )


    return input_tensor


# ============================================================
# MULTI-LABEL PREDICTION
# ============================================================

def predict_multi_label(

    model,
    input_tensor,
    threshold

):

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

                    MULTI_LABEL_CLASSES[
                        index
                    ],

                "probability":

                    float(probability)

            })


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

    model,
    input_tensor,
    top_k=5

):

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

                SINGLE_LABEL_CLASSES[
                    index
                ],

            "probability":

                float(probability)

        })


    return predictions


# ============================================================
# MAIN
# ============================================================

def main():


    parser = argparse.ArgumentParser(

        description=

        "Hybrid Indian Food Prediction"

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

        default=5

    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    print(

        "\nLoading Multi-Label Model..."

    )


    multi_model = (

        load_multi_label_model(

            checkpoint_path=

            args.multi_checkpoint,

            device=device

        )

    )


    print(

        "Multi-Label Model Loaded!"

    )


    print(

        "\nLoading Single-Label Model..."

    )


    single_model = (

        load_single_label_model(

            checkpoint_path=

            args.single_checkpoint,

            device=device

        )

    )


    print(

        "Single-Label Model Loaded!"

    )


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    print(

        "\nProcessing Image..."

    )


    input_tensor = prepare_image(

        args.image,

        device

    )


    # ========================================================
    # MULTI-LABEL PREDICTION
    # ========================================================

    multi_predictions = (

        predict_multi_label(

            model=multi_model,

            input_tensor=input_tensor,

            threshold=args.threshold

        )

    )


    # ========================================================
    # SINGLE-LABEL PREDICTION
    # ========================================================

    single_predictions = (

        predict_single_label(

            model=single_model,

            input_tensor=input_tensor,

            top_k=args.top_k

        )

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(

        "\n"

        + "=" * 60

    )


    print(

        "MULTI-LABEL FOOD DETECTION"

    )


    print(

        "=" * 60

    )


    if len(
        multi_predictions
    ) == 0:


        print(

            "\nNo foods detected above threshold."

        )


    else:


        print()


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


    # ========================================================
    # SINGLE LABEL RESULTS
    # ========================================================

    print(

        "\n"

        + "=" * 60

    )


    print(

        "SINGLE-LABEL FOOD VERIFICATION"

    )


    print(

        "=" * 60

        + "\n"

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


    # ========================================================
    # BEST HYBRID RESULT
    # ========================================================

    print(

        "\n"

        + "=" * 60

    )


    print(

        "HYBRID SUMMARY"

    )


    print(

        "=" * 60

        + "\n"

    )


    print(

        "Multi-label detected foods: "

        f"{len(multi_predictions)}"

    )


    print(

        "\nBest single-label prediction:"

    )


    best_single = (

        single_predictions[0]

    )


    print(

        f"{best_single['class']} "

        f"({best_single['probability'] * 100:.2f}%)"

    )


    print(

        "\nDone!"

    )


if __name__ == "__main__":

    main()