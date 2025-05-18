# -*- coding: utf-8 -*-
# file: asgcn.py
# author:  <gene_zhangchen@163.com>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.dynamic_rnn import DynamicLSTM
from pyabsa.networks.sa_encoder import Encoder


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASGCN_BERT_Unit(nn.Module):
    def __init__(self, bert, config):
        super(ASGCN_BERT_Unit, self).__init__()
        self.config = config
        self.embed = bert
        self.text_lstm = DynamicLSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gc1 = GraphConvolution(2 * config.hidden_dim, 2 * config.hidden_dim)
        self.gc2 = GraphConvolution(2 * config.hidden_dim, 2 * config.hidden_dim)
        self.text_embed_dropout = nn.Dropout()

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        min_len = torch.inf
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
            min_len = len(weight[i]) if len(weight[i]) < min_len else min_len
        weight = [w[:seq_len] for w in weight]
        weight = (
            torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.config.device)
        )
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        min_len = torch.inf
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = [m[:seq_len] for m in mask]
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.config.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1
        )
        text = self.embed(text_indices)["last_hidden_state"]
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(
            self.gc1(
                self.position_weight(text_out, aspect_double_idx, text_len, aspect_len),
                adj,
            )
        )
        x = F.relu(
            self.gc2(
                self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj
            )
        )
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim

        return x


class ASGCN_BERT(nn.Module):
    inputs = [
        "text_indices",
        "aspect_indices",
        "left_indices",
        "dependency_graph",
        "left_aspect_indices",
        "left_left_indices",
        "left_dependency_graph",
        "right_aspect_indices",
        "right_left_indices",
        "right_dependency_graph",
    ]

    def __init__(self, bert, config):
        super(ASGCN_BERT, self).__init__()
        self.config = config
        self.asgcn_left = ASGCN_BERT_Unit(bert, config) if self.config.lsa else None
        self.asgcn_central = ASGCN_BERT_Unit(bert, config)
        self.encoder = Encoder(bert.config, config)
        self.dropout = nn.Dropout(config.dropout)
        self.pooler = BertPooler(bert.config)
        self.asgcn_right = ASGCN_BERT_Unit(bert, config) if self.config.lsa else None
        self.linear = nn.Linear(self.config.hidden_dim * 6, self.config.output_dim)
        self.dense = nn.Linear(self.config.hidden_dim * 2, self.config.output_dim)

    def forward(self, inputs):
        res = {"logits": None}
        if self.config.lsa:
            cat_feat = torch.cat(
                (
                    self.asgcn_left(
                        [
                            inputs["text_indices"],
                            inputs["left_aspect_indices"],
                            inputs["left_left_indices"],
                            inputs["left_dependency_graph"],
                        ]
                    ),
                    self.asgcn_central(
                        [
                            inputs["text_indices"],
                            inputs["aspect_indices"],
                            inputs["left_indices"],
                            inputs["dependency_graph"],
                        ]
                    ),
                    self.asgcn_right(
                        [
                            inputs["text_indices"],
                            inputs["right_aspect_indices"],
                            inputs["right_left_indices"],
                            inputs["right_dependency_graph"],
                        ]
                    ),
                ),
                -1,
            )
            cat_feat = self.dropout(cat_feat)
            res["logits"] = self.linear(cat_feat)
        else:
            cat_feat = self.asgcn_central(
                [
                    inputs["text_indices"],
                    inputs["aspect_indices"],
                    inputs["left_indices"],
                    inputs["dependency_graph"],
                ]
            )
            cat_feat = self.dropout(cat_feat)
            res["logits"] = self.dense(cat_feat)

        return res
