# -*- coding: utf-8 -*-
# file: MAELoss.py
# time: 2022/11/24 20:11

import torch
from torch import nn


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))
