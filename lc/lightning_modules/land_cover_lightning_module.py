# -*- coding: utf-8 -*-
import argparse
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from lc.lightning_modules import metrics
from lc.models.unet import UNet


class LandCoverLightningModule(pl.LightningModule):
    """
    LightningModule for training the Land Cover Segmentation Model.
    """

    def __init__(self, **kwargs):
        super(LandCoverLightningModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.model = self.build_model()

        # loss
        self.loss = torch.nn.CrossEntropyLoss()

        # metrics
        self.iou_train = metrics.Iou(num_classes=self.hparams.num_classes)
        self.iou_val = metrics.Iou(num_classes=self.hparams.num_classes)

    def build_model(self) -> nn.Module:
        if self.hparams.architecture == "unet":
            return UNet(num_classes=self.hparams.num_classes)

        raise NotImplementedError(
            f"Architecture '{self.hparams.architecture}' is not supported."
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        image, mask = batch["image"], batch["mask"]
        preds = self(image)
        loss = self.loss(preds, mask)
        preds = preds.argmax(dim=1)
        self.iou_train(preds, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Dict[str, Union[float, int]]:
        image, mask = batch["image"], batch["mask"]
        preds = self(image)
        loss = self.loss(preds, mask)
        preds = preds.argmax(dim=1)
        self.iou_val(preds, mask)
        return {"val/loss": loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # Compute and log metrics across epoch
        metrics_avg = self.iou_train.compute()
        self.log("train/mIoU", metrics_avg.miou)
        self.log("train/accuracy", metrics_avg.accuracy.mean())
        self.log("train/precision", metrics_avg.precision.mean())
        self.log("train/recall", metrics_avg.recall.mean())
        self.log("train/specificity", metrics_avg.specificity.mean())
        self.iou_train.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        # Compute and log metrics across epoch
        loss_mean = torch.stack([output["val/loss"] for output in outputs]).mean()

        metrics_avg = self.iou_val.compute()
        self.log("val/loss", loss_mean)
        self.log("hp_metric", metrics_avg.miou)
        self.log("val/mIoU", metrics_avg.miou)
        self.log("val/accuracy", metrics_avg.accuracy.mean())
        self.log("val/precision", metrics_avg.precision.mean())
        self.log("val/recall", metrics_avg.recall.mean())
        self.log("val/specificity", metrics_avg.specificity.mean())
        self.iou_val.reset()

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[str, Any]]]]:
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.hparams.max_epochs,
            pct_start=self.hparams.pct_start,
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(
            parents=[parent], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--max_lr",
            default=1e-3,
            type=float,
            help="The max learning rate for the 1Cycle LR Scheduler",
        )
        parser.add_argument(
            "--pct_start",
            default=0.3,
            type=Union[float, int],
            help="The percentage of the cycle (in number of steps) spent increasing the learning rate",
        )
        parser.add_argument(
            "--div_factor",
            default=2,
            type=float,
            help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
        )
        parser.add_argument(
            "--final_div_factor",
            default=1e2,
            type=float,
            help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
        )
        return parser
