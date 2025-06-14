# -*- coding: utf-8 -*-
# file: bert_base.py

# Copyright (C) 2020. All Rights Reserved.

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, bert, config):
        super(BERT, self).__init__()
        self.bert = bert
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.embed_dim, config.output_dim)

    def forward(self, inputs):
        text_indices = inputs["text_indices"]
        text_features = self.bert(text_indices)["last_hidden_state"]
        pooled_output = self.pooler(text_features)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return {"logits": logits, "hidden_state": pooled_output}
