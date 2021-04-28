import torch
from torch import nn, Tensor
from conformer.models.activations import Swish
from conformer.models.attention import MultiHeadAttentionWithRelativePositionalEmbedding
from conformer.models.convolutions import PointwiseConv1d, DepthwiseConv1d


class MultiHeadedSelfAttentionModule(nn.Module):
    """ Multi-Head Self Attention Module """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        self.layer_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttentionWithRelativePositionalEmbedding(dim, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        norm_val = self.layer_norm(inputs)
        # TODO:: Add Positional Embedding
        attention = self.attention(norm_val, norm_val, norm_val)
        output = self.dropout(attention)
        return output


class ConvolutionModule(nn.Module):
    """ Convolution Module """

    def __init__(
            self,
            in_channels: int,
            expansion_factor: int = 2,
            kernel_size: int = 31,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for padding"

        self.device = device
        first_channels = in_channels * expansion_factor
        second_channels = first_channels // 2

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(1, 2),
            PointwiseConv1d(in_channels, first_channels),
            nn.GLU(dim=1),
            DepthwiseConv1d(second_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(second_channels),
            Swish(),
            PointwiseConv1d(second_channels, in_channels),
            Transpose(1, 2),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs.to(self.device))


class FeedForwardModule(nn.Module):
    """ Feed Forward Module """

    def __init__(
            self,
            dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device
        inner_dim = dim * expansion_factor

        self.sequential = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs.to(self.device))


class ResidualModule(nn.Module):
    """ Residual Module """

    def __init__(self, module: nn.Module, factor: float = 1.0):
        super().__init__()
        self.module = module
        self.factor = factor

    def forward(self, inputs: Tensor) -> Tensor:
        module_output = self.module(inputs) * self.factor
        return module_output + inputs


class Transpose(nn.Module):
    def __init__(
            self,
            dim0: int = 0,
            dim1: int = 1,
    ):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(self.dim0, self.dim1)
