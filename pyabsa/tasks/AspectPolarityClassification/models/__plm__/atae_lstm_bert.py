# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from pyabsa.networks.attention import NoQueryAttention
from pyabsa.networks.dynamic_rnn import DynamicLSTM
from pyabsa.networks.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM_BERT(nn.Module):
    inputs = ["text_indices", "aspect_indices"]

    def __init__(self, bert, config):
        super(ATAE_LSTM_BERT, self).__init__()
        self.config = config
        self.embed = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(
            config.embed_dim * 2, config.hidden_dim, num_layers=1, batch_first=True
        )
        self.attention = NoQueryAttention(
            config.hidden_dim + config.embed_dim, score_function="bi_linear"
        )
        self.dense = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs["text_indices"], inputs["text_indices"]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)["last_hidden_state"]
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)["last_hidden_state"]
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        return {"logits": out}
