import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(train=True):

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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
# MULTI-LABEL DATASET
# ============================================================

class MultiLabelThaliDataset(Dataset):

    def __init__(
        self,
        data_dir,
        mapping_path,
        transform=None
    ):

        self.data_dir = Path(data_dir)

        self.images_dir = (
            self.data_dir / "train" / "images"
        )

        self.masks_dir = (
            self.data_dir / "train" / "masks"
        )

        self.transform = transform


        # ----------------------------------------------------
        # LOAD CLASS MAPPING
        # ----------------------------------------------------

        with open(mapping_path, "r") as f:
            mapping = json.load(f)


        # Remove background (class 0)

        self.class_mapping = {
            int(k): v
            for k, v in mapping.items()
            if int(k) != 0
        }


        # Food class names

        self.class_names = [
            self.class_mapping[i]
            for i in sorted(self.class_mapping.keys())
        ]


        self.num_classes = len(self.class_names)


        # ----------------------------------------------------
        # FIND MATCHING IMAGE + MASK PAIRS
        # ----------------------------------------------------

        self.samples = self._load_samples()


        print("\nDataset loaded successfully!")
        print(f"Matched image-mask pairs: {len(self.samples)}")
        print(f"Number of food classes: {self.num_classes}")


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
    # LOAD MATCHED PAIRS
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
            f"Images found: {len(image_files)}"
        )

        print(
            f"Masks found: {len(mask_files)}"
        )

        print(
            f"Matched pairs: {len(samples)}"
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

        image_path, mask_path = (
            self.samples[index]
        )


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

        mask = Image.open(mask_path)

        mask = np.array(mask)


        # Get unique food IDs

        unique_ids = np.unique(mask)


        # ----------------------------------------------------
        # CREATE MULTI-HOT LABEL
        # ----------------------------------------------------

        label = np.zeros(
            self.num_classes,
            dtype=np.float32
        )


        for class_id in unique_ids:

            class_id = int(class_id)


            # Ignore background

            if class_id == 0:
                continue


            # Convert dataset ID → multi-label index

            if class_id in self.class_mapping:

                label_index = class_id - 1

                label[label_index] = 1.0


        label = torch.tensor(
            label,
            dtype=torch.float32
        )


        return image, label


# ============================================================
# BUILD DATASETS
# ============================================================

def build_multilabel_datasets(
    data_dir,
    mapping_path,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):


    # --------------------------------------------------------
    # FULL DATASET
    # --------------------------------------------------------

    full_dataset = MultiLabelThaliDataset(

        data_dir=data_dir,

        mapping_path=mapping_path,

        transform=get_transforms(train=False)

    )


    total_size = len(full_dataset)


    train_size = int(
        train_ratio * total_size
    )

    val_size = int(
        val_ratio * total_size
    )

    test_size = (
        total_size
        -
        train_size
        -
        val_size
    )


    generator = torch.Generator()

    generator.manual_seed(seed)


    train_dataset, val_dataset, test_dataset = random_split(

        full_dataset,

        [
            train_size,
            val_size,
            test_size
        ],

        generator=generator

    )


    # --------------------------------------------------------
    # IMPORTANT:
    # DIFFERENT TRANSFORMS FOR TRAINING
    # --------------------------------------------------------

    train_dataset.dataset.transform = (
        get_transforms(train=True)
    )


    return (

        train_dataset,

        val_dataset,

        test_dataset,

        full_dataset.class_names

    )