from pathlib import Path
import json

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset, random_split
from torchvision import transforms


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

def get_transforms(train=True):

    if train:

        return transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:

        return transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    mapping_path = Path(
        r"C:\Users\Gaddam Sai Teja\Indian_Thali_ICVGIP2025"
    ) / "food_scanner" / "seg_full_final.json"


    if not mapping_path.exists():

        raise FileNotFoundError(
            f"Class mapping file not found:\n{mapping_path}"
        )


    with open(mapping_path, "r", encoding="utf-8") as f:

        class_mapping = json.load(f)


    # Ignore background (ID = 0)
    class_ids = list(range(1, 51))

    class_names = [

        class_mapping[str(class_id)]

        for class_id in class_ids

    ]


    return class_ids, class_names


# ============================================================
# MULTI-LABEL DATASET
# ============================================================

class MultiLabelFoodDataset(Dataset):


    def __init__(self, data_dir, transform=None):


        self.data_dir = Path(data_dir)


        self.images_dir = (

            self.data_dir /
            "train" /
            "images"

        )


        self.masks_dir = (

            self.data_dir /
            "train" /
            "masks"

        )


        self.transform = transform


        # ----------------------------------------------------
        # CHECK FOLDERS
        # ----------------------------------------------------

        if not self.images_dir.exists():

            raise FileNotFoundError(

                f"Images folder not found:\n"
                f"{self.images_dir}"

            )


        if not self.masks_dir.exists():

            raise FileNotFoundError(

                f"Masks folder not found:\n"
                f"{self.masks_dir}"

            )


        # ----------------------------------------------------
        # LOAD REAL CLASS NAMES
        # ----------------------------------------------------

        self.class_ids, self.class_names = load_class_names()


        # ----------------------------------------------------
        # LOAD MATCHED IMAGE-MASK PAIRS
        # ----------------------------------------------------

        self.samples = self._load_samples()


        if len(self.samples) == 0:

            raise RuntimeError(
                "No matching image-mask pairs found."
            )


        print("\nDataset loaded successfully!")

        print(
            f"Matched image-mask pairs: "
            f"{len(self.samples)}"
        )

        print(
            f"Number of classes: "
            f"{len(self.class_names)}"
        )


    # ========================================================
    # IMAGE KEY
    # ========================================================

    def _get_image_key(self, filename):

        stem = Path(filename).stem

        return stem.replace(
            "_leftImg8bit",
            ""
        )


    # ========================================================
    # MASK KEY
    # ========================================================

    def _get_mask_key(self, filename):

        stem = Path(filename).stem

        return stem.replace(
            "_gtFine_labelIds",
            ""
        )


    # ========================================================
    # MATCH FILES
    # ========================================================

    def _load_samples(self):


        image_files = {}


        for image_path in self.images_dir.iterdir():

            if image_path.suffix.lower() in [

                ".jpg",
                ".jpeg",
                ".png"

            ]:


                key = self._get_image_key(
                    image_path.name
                )


                image_files[key] = image_path


        mask_files = {}


        for mask_path in self.masks_dir.iterdir():

            if mask_path.suffix.lower() == ".png":


                key = self._get_mask_key(
                    mask_path.name
                )


                mask_files[key] = mask_path


        # ----------------------------------------------------
        # FIND COMMON FILES
        # ----------------------------------------------------

        common_keys = sorted(

            set(image_files.keys())

            &

            set(mask_files.keys())

        )


        samples = []


        for key in common_keys:


            samples.append(

                (

                    image_files[key],

                    mask_files[key]

                )

            )


        print(
            f"\nImages found: "
            f"{len(image_files)}"
        )

        print(
            f"Masks found: "
            f"{len(mask_files)}"
        )

        print(
            f"Matched pairs: "
            f"{len(samples)}"
        )


        return samples


    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):


        image_path, mask_path = self.samples[index]


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)


        # ----------------------------------------------------
        # LOAD MASK
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        )


        mask = np.array(mask)


        # ----------------------------------------------------
        # FIND UNIQUE LABELS
        # ----------------------------------------------------

        present_labels = np.flatnonzero(
            np.bincount(mask.ravel(), minlength=51)
        )


        # ----------------------------------------------------
        # CREATE MULTI-HOT VECTOR
        #
        # 50 outputs:
        #
        # index 0  -> class ID 1
        # index 1  -> class ID 2
        # ...
        # index 49 -> class ID 50
        # ----------------------------------------------------

        label_vector = np.zeros(

            len(self.class_ids),

            dtype=np.float32

        )


        for label in present_labels:


            # Ignore background

            if label == 0:

                continue


            # Food class IDs = 1 to 50

            if int(label) in self.class_ids:


                label_vector[
                    int(label) - 1
                ] = 1.0


        label_vector = torch.tensor(

            label_vector,

            dtype=torch.float32

        )


        return image, label_vector


# ============================================================
# BUILD DATASETS
# ============================================================

def build_datasets(

    data_dir="data",

    val_size=0.15,

    test_size=0.15,

    seed=42

):


    # --------------------------------------------------------
    # BASE DATASET
    # --------------------------------------------------------

    full_dataset = MultiLabelFoodDataset(

        data_dir=data_dir,

        transform=None

    )


    total_size = len(full_dataset)


    test_length = int(
        total_size * test_size
    )


    val_length = int(
        total_size * val_size
    )


    train_length = (

        total_size

        -

        val_length

        -

        test_length

    )


    generator = torch.Generator().manual_seed(
        seed
    )


    train_subset, val_subset, test_subset = random_split(

        full_dataset,

        [

            train_length,

            val_length,

            test_length

        ],

        generator=generator

    )


    # --------------------------------------------------------
    # TRAIN DATASET
    # --------------------------------------------------------

    train_dataset = MultiLabelFoodDataset(

        data_dir=data_dir,

        transform=get_transforms(train=True)

    )


    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    val_dataset = MultiLabelFoodDataset(

        data_dir=data_dir,

        transform=get_transforms(train=False)

    )


    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_dataset = MultiLabelFoodDataset(

        data_dir=data_dir,

        transform=get_transforms(train=False)

    )


    # --------------------------------------------------------
    # USE SAME SPLIT INDICES
    # --------------------------------------------------------

    train_dataset.samples = [

        full_dataset.samples[i]

        for i in train_subset.indices

    ]


    val_dataset.samples = [

        full_dataset.samples[i]

        for i in val_subset.indices

    ]


    test_dataset.samples = [

        full_dataset.samples[i]

        for i in test_subset.indices

    ]


    return (

        train_dataset,

        val_dataset,

        test_dataset,

        full_dataset.class_names

    )


