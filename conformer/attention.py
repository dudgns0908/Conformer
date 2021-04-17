import numpy as np
import torch
from torch import nn, Tensor


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scaled_matmul = torch.bmm(query, torch.transpose(key, 1, 2)) / self.dim
        softmax = F.softmax(scaled_matmul)
        attention = torch.bmm(softmax, value)

        return attention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(
            self,
            dim: int = 512,
            num_heads: int = 8,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dk = dim // num_heads

        self.query_projection = nn.Linear(dim, dim * self.dk)
        self.key_projection = nn.Linear(dim, dim * self.dk)
        self.value_projection = nn.Linear(dim, dim * self.dk)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dim=self.dk)
        self.linear = nn.Linear(self.dk * self.num_heads, dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size = query.shape[0]

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2).contiguous()
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2).contiguous()
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2).contiguous()

        attention = self.scaled_dot_product_attention(query, key, value)
        concat = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dk)
        linear = self.linear(concat)

        return linear
