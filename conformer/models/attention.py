from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from conformer.models.embedding import PositionalEncoding


class MultiHeadAttentionWithRelativePositionalEmbedding(nn.Module):
    # TODO:: Read paper that 'transformer XL' and understand it.
    def __init__(
            self,
            dim: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "d_model % num_heads should be zero."

        self.dim = dim
        self.d_head = dim // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = np.sqrt(dim)

        self.positional_encoding = PositionalEncoding(d_model=dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.pos_projection = nn.Linear(dim, dim, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.dropout = nn.Dropout(p=dropout_p)
        self.out_projection = nn.Linear(dim, dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_length, _ = value.size()

        # Positional embedding (U)
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        pos_embedding = self.pos_projection(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        # Q, K, V
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        positional_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        positional_score = self._relative_shift(positional_score)
        score = (content_score + positional_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim)

        return self.out_projection(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


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

        self.query_projection = nn.Linear(dim, num_heads * self.dk)
        self.key_projection = nn.Linear(dim, num_heads * self.dk)
        self.value_projection = nn.Linear(dim, num_heads * self.dk)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dim=self.dk)
        self.linear = nn.Linear(self.dk * self.num_heads, dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size = query.shape[0]

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dk)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.dk)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dk)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dk)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dk)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dk)

        attention = self.scaled_dot_product_attention(query, key, value)
        attention = attention.view(self.num_heads, batch_size, -1, self.dk)
        concat = attention.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dk)
        linear = self.linear(concat)

        return linear
