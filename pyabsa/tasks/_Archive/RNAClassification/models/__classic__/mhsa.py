# -*- coding: utf-8 -*-
# file: mhsa.py
# time: 31/10/2022 20:00

import torch
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.sa_encoder import Encoder
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, bert_config, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = config
        self.config.hidden_size = self.config.hidden_dim
        self.mhsa = Encoder(
            bert_config=bert_config,
            config=self.config,
            layer_num=self.config.num_mhsa_layer,
        )
        self.config = config

    def forward(self, x):
        return self.mhsa(x)


class MHSA(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, embedding_matrix, config):
        super(MHSA, self).__init__()
        self.config = config
        self.bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        self.bert_config.hidden_size = self.config.hidden_dim
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.mhsa = MultiHeadSelfAttention(self.bert_config, self.config)
        self.pooler = BertPooler(self.bert_config)

        self.dense = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        out = self.mhsa(x)
        out = self.pooler(out)
        out = self.dense(out)
        return out
