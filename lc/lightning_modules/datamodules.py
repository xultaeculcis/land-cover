# -*- coding: utf-8 -*-
import logging
import os
from argparse import ArgumentParser
from glob import glob
from typing import Dict, Optional, Union, Tuple

import albumentations as a
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
os.environ["NUMEXPR_MAX_THREADS"] = "16"


class LandCoverDataset(Dataset):
    def __init__(self, images, masks, augmentations, stage):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations
        self.stage = stage
        self.common_transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index) -> Dict[str, Union[Tensor, list]]:
        image = self._read_image(self.images[index])
        mask = self._read_mask(self.masks[index])

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return {
            "image": self.common_transform(image),
            "mask": self.common_transform(mask).long(),
        }

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _read_mask(label_path: str) -> np.ndarray:
        mask = cv2.imread(
            label_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
        )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

        if len(mask.shape) != 2:
            raise RuntimeError(f"The shape of label must be (H, W). Got: {mask.shape}")

        return mask.astype(np.int32)

    @staticmethod
    def _read_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img.shape) != 3:
            raise RuntimeError(
                f"The shape of image must be (H, W, C). Got: {img.shape}"
            )

        return img


class LandCoverDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        resize: Optional[Tuple[int, int]] = (256, 256),
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 4,
        seed: Optional[int] = 42,
    ):
        super(LandCoverDataModule, self).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.resize_width, self.resize_height = resize

        self.class_dict_df = pd.read_csv(os.path.join(self.data_path, "class_dict.csv"))
        self.class_codes = self.class_dict_df["name"].values
        self.class_dict_df["pixel_value"] = round(
            self.class_dict_df["r"] * 299 / 1000
            + self.class_dict_df["g"] * 587 / 1000
            + self.class_dict_df["b"] * 114 / 1000,
            0,
        ).astype(int, copy=False)

        train_images = sorted(glob(os.path.join(self.data_path, "train/*.jpg")))
        train_masks = sorted(glob(os.path.join(self.data_path, "train/*.png")))

        val_images = sorted(glob(os.path.join(self.data_path, "valid/*.jpg")))
        val_masks = sorted(glob(os.path.join(self.data_path, "valid/*.png")))

        test_images = sorted(glob(os.path.join(self.data_path, "test/*.jpg")))
        test_masks = sorted(glob(os.path.join(self.data_path, "test/*.png")))

        logging.info(
            f"Train/Validation/Test split sizes (images): {len(train_images)}/{len(val_images)}/{len(test_images)}"
        )
        logging.info(
            f"Train/Validation/Test split sizes (masks): {len(train_masks)}/{len(val_masks)}/{len(test_masks)}"
        )

        self.train_dataset = LandCoverDataset(
            images=train_images,
            masks=train_masks,
            augmentations=a.Compose(
                [
                    a.HorizontalFlip(),
                    a.VerticalFlip(),
                    a.RandomRotate90(),
                    a.RandomCrop(height=self.resize_height, width=self.resize_width),
                ]
            ),
            stage="train",
        )

        self.val_dataset = LandCoverDataset(
            images=val_images,
            masks=val_masks,
            augmentations=a.Compose(
                [
                    a.CenterCrop(height=self.resize_height, width=self.resize_width),
                ]
            ),
            stage="val",
        )

        self.test_dataset = LandCoverDataset(
            images=test_images,
            masks=test_masks,
            augmentations=a.Compose(
                [
                    a.CenterCrop(height=self.resize_height, width=self.resize_width),
                ]
            ),
            stage="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds datamodule specific arguments.

        :param parent_parser: The parent parser.
        :returns: The parser.
        """
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default="../datasets/",
        )
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--resize", type=Tuple[int, int], default=(256, 256))
        parser.add_argument("--num_classes", type=int, default=7)
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    args = LandCoverDataModule.add_data_specific_args(parser).parse_args()
    args.data_path = "../../datasets"

    dm = LandCoverDataModule(
        data_path=args.data_path,
        resize=args.resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    dl = dm.train_dataloader()

    def matplotlib_imshow(batch):
        # create grid of images
        img_grid = torchvision.utils.make_grid(
            batch, nrow=4, normalize=False, padding=0
        )
        # show images
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        image = batch["image"]
        mask = batch["mask"]

        matplotlib_imshow(image)
        matplotlib_imshow(mask.unsqueeze(1))
        break
