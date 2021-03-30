# -*- coding: utf-8 -*-
import torch

from lc.lightning_modules.metrics import Iou


def test_iou():
    # Arrange
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    pred = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    label[:, :3, :3] = 1
    pred[:, -3:, -3:] = 1
    expected_iou = torch.tensor([2.0 / 12, 4.0 / 14], device=device)

    # Act
    iou_train = Iou(num_classes=2)
    iou_train.to(device)
    iou_train(pred, label)
    metrics_r = iou_train.compute()

    # Assert
    iou_per_class = metrics_r.iou_per_class
    assert (iou_per_class - expected_iou).sum() < 1e-6
