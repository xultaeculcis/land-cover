# -*- coding: utf-8 -*-
from typing import Tuple, Optional
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import numpy as np


logging.basicConfig(level=logging.INFO)


def make_patches(img_path: str, patch_shape: Optional[Tuple[int, int]] = (512, 512)):
    pass


def train_val_split(df: pd.DataFrame, out_path: str):
    logging.info("Running train/val split")
    X = df[df["split"] == "train"]

    logging.info(f"After filtering images without masks: {len(X)}")
    dummy_y = np.arange(len(X))

    X_train, X_test, _, _ = train_test_split(
        X, dummy_y, test_size=0.13, random_state=42
    )

    logging.info(f"Train/Val size: {len(X_train)}/{len(X_test)}")
    X_train.to_csv(os.path.join(out_path, "train.csv"), header=True, index=False)
    X_test.to_csv(os.path.join(out_path, "val.csv"), header=True, index=False)


if __name__ == "__main__":
    data_path = "../../datasets/"
    frame = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    logging.info(f"Total rows: {len(frame)}")
    train_val_split(frame, data_path)
