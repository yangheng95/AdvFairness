
import torch
from torch import nn


class BERTMeanPooler(nn.Module):
    def __init__(
            self,
    ):
        super(BERTMeanPooler, self).__init__()

    def forward(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
