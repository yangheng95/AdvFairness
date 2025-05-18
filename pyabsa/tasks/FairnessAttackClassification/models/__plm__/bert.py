# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT_MLP(nn.Module):
    inputs = ["author_indices", "news_indices"]

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        author_indices = inputs[0]
        text_raw_indices = inputs[1]
        last_hidden_state1 = self.bert(author_indices)["last_hidden_state"]
        last_hidden_state2 = self.bert(text_raw_indices)["last_hidden_state"]
        last_hidden_state = self.linear(torch.cat((last_hidden_state1, last_hidden_state2), dim=-1))

        pooled_out = self.pooler(last_hidden_state)
        out = self.dense(pooled_out)
        return out
