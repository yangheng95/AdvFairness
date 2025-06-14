# -*- coding: utf-8 -*-
# file: bert.py
# time: 02/11/2022 15:48



import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT_MLP(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.config = config
        self.bert = bert
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.sigmoid_regression:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        last_hidden_state = self.bert(text_raw_indices)["last_hidden_state"]
        pooled_out = self.pooler(last_hidden_state)
        pooled_out = self.dropout(pooled_out)
        pooled_out = self.tanh(pooled_out)
        out = self.dense(pooled_out)
        if self.sigmoid:
            out = self.sigmoid(out)
        return out
