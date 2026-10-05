from pathlib import Path

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset, random_split
from torchvision import transforms


# ============================================================
# SEGMENTATION DATASET
# ============================================================

class FoodSegmentationDataset(Dataset):

    def __init__(
        self,
        data_dir="data",
        image_size=512
    ):

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

        self.image_size = image_size


        # ----------------------------------------------------
        # CHECK DIRECTORIES
        # ----------------------------------------------------

        if not self.images_dir.exists():

            raise FileNotFoundError(

                f"Images directory not found: "

                f"{self.images_dir}"

            )


        if not self.masks_dir.exists():

            raise FileNotFoundError(

                f"Masks directory not found: "

                f"{self.masks_dir}"

            )


        # ----------------------------------------------------
        # LOAD MATCHED SAMPLES
        # ----------------------------------------------------

        self.samples = self._load_samples()


        if len(self.samples) == 0:

            raise RuntimeError(

                "No matching image-mask pairs found."

            )


        print(

            f"\nSegmentation dataset loaded!"

        )

        print(

            f"Matched pairs: "

            f"{len(self.samples)}"

        )


    # ========================================================
    # IMAGE KEY
    # ========================================================

    def _image_key(self, filename):

        return (

            Path(filename)

            .stem

            .replace(

                "_leftImg8bit",

                ""

            )

        )


    # ========================================================
    # MASK KEY
    # ========================================================

    def _mask_key(self, filename):

        return (

            Path(filename)

            .stem

            .replace(

                "_gtFine_labelIds",

                ""

            )

        )


    # ========================================================
    # LOAD SAMPLES
    # ========================================================

    def _load_samples(self):

        image_files = {}


        for path in self.images_dir.iterdir():

            if path.suffix.lower() in [

                ".jpg",

                ".jpeg",

                ".png"

            ]:

                key = self._image_key(

                    path.name

                )


                image_files[key] = path


        mask_files = {}


        for path in self.masks_dir.iterdir():

            if path.suffix.lower() == ".png":

                key = self._mask_key(

                    path.name

                )


                mask_files[key] = path


        common_keys = sorted(

            set(image_files)

            &

            set(mask_files)

        )


        samples = [

            (

                image_files[key],

                mask_files[key]

            )

            for key in common_keys

        ]


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
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(

            self.samples

        )


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(

        self,

        index

    ):


        image_path, mask_path = (

            self.samples[index]

        )


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(

            image_path

        ).convert(

            "RGB"

        )


        # ----------------------------------------------------
        # LOAD MASK
        # ----------------------------------------------------

        mask = Image.open(

            mask_path

        )


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = transforms.functional.resize(

            image,

            (

                self.image_size,

                self.image_size

            ),

            interpolation=

            transforms.InterpolationMode.BILINEAR

        )


        # ----------------------------------------------------
        # RESIZE MASK
        #
        # IMPORTANT:
        # NEAREST interpolation preserves class IDs
        # ----------------------------------------------------

        mask = transforms.functional.resize(

            mask,

            (

                self.image_size,

                self.image_size

            ),

            interpolation=

            transforms.InterpolationMode.NEAREST

        )


        # ----------------------------------------------------
        # IMAGE → TENSOR
        # ----------------------------------------------------

        image = transforms.functional.to_tensor(

            image

        )


        image = transforms.functional.normalize(

            image,

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


        # ----------------------------------------------------
        # MASK → LONG TENSOR
        # ----------------------------------------------------

        mask = np.array(

            mask,

            dtype=np.int64

        )


        mask = torch.from_numpy(

            mask

        ).long()


        return (

            image,

            mask

        )


# ============================================================
# BUILD DATASETS
# ============================================================

def build_segmentation_datasets(

    data_dir="data",

    image_size=512,

    val_size=0.15,

    seed=42

):


    dataset = FoodSegmentationDataset(

        data_dir=data_dir,

        image_size=image_size

    )


    total_size = len(

        dataset

    )


    val_length = int(

        total_size * val_size

    )


    train_length = (

        total_size

        -

        val_length

    )


    generator = (

        torch.Generator()

        .manual_seed(seed)

    )


    train_dataset, val_dataset = (

        random_split(

            dataset,

            [

                train_length,

                val_length

            ],

            generator=generator

        )

    )


    return (

        train_dataset,

        val_dataset

    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":


    train_dataset, val_dataset = (

        build_segmentation_datasets()

    )


    print(

        "\n=============================="

    )

    print(

        "SEGMENTATION DATASET TEST"

    )

    print(

        "=============================="

    )


    print(

        f"\nTraining samples: "

        f"{len(train_dataset)}"

    )


    print(

        f"Validation samples: "

        f"{len(val_dataset)}"

    )


    image, mask = train_dataset[0]


    print(

        f"\nImage shape: "

        f"{image.shape}"

    )


    print(

        f"Mask shape: "

        f"{mask.shape}"

    )


    print(

        f"Mask labels: "

        f"{torch.unique(mask)}"

    )