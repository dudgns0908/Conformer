import math

import numpy as np
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """ Positional Embedding """

    def __init__(
            self,
            d_model: int = 512,
            max_len: int = 10000,
    ) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelativePositionalEncoding(nn.Module):
    """ Relative Positional Encoding """

    def __init__(self):
        super(RelativePositionalEncoding, self).__init__()

