# -*- coding: utf-8 -*-
# file: R2Loss.py
# time: 2022/11/24 20:06

import torch
from torch import nn


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true, dim=0)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        r2 = 1 - ss_res / ss_tot
        return 1 - torch.mean(r2)
