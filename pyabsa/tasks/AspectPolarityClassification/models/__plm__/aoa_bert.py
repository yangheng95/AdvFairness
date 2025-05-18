# -*- coding: utf-8 -*-
# file: aoa.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyabsa.networks.dynamic_rnn import DynamicLSTM


class AOA_BERT(nn.Module):
    inputs = [
        "text_indices",
        "aspect_indices",
        "left_text_indices",
        "left_aspect_indices",
        "right_text_indices",
        "right_aspect_indices",
    ]

    def __init__(self, bert, config):
        super(AOA_BERT, self).__init__()
        self.config = config
        self.embed = bert
        self.ctx_lstm = DynamicLSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.asp_lstm = DynamicLSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dense = nn.Linear(2 * config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        text_indices = inputs["text_indices"]  # batch_size x seq_len
        aspect_indices = inputs["aspect_indices"]  # batch_size x seq_len
        ctx_len = torch.sum(text_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_indices)[
            "last_hidden_state"
        ]  # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices)[
            "last_hidden_state"
        ]  # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(
            ctx, ctx_len
        )  # batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(
            asp, asp_len
        )  # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = torch.matmul(
            ctx_out, torch.transpose(asp_out, 1, 2)
        )  # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(
            interaction_mat, dim=1
        )  # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(
            interaction_mat, dim=2
        )  # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True)  # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(
            alpha, beta_avg.transpose(1, 2)
        )  # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(
            -1
        )  # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum)  # batch_size x polarity_dim

        return {"logits": out}
