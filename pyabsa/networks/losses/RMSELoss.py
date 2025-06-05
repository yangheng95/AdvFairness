# -*- coding: utf-8 -*-
# file: RMSELoss.py
# time: 2022/11/24 20:10
  
import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.MSELoss()(y_pred, y_true))
