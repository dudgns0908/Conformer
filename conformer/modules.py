import torch
from torch import nn, Tensor
from conformer.activations import Swish


class MultiHeadedSelfAttentionModule(nn.Module):
    """ Multi-Head Self Attention """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        self.P = 2 ** 12
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, num_heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, num_heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((num_heads, self.d_k)), requires_grad=True)

        self.sequential = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttention(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs.to(self.device))


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        self.sequential = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttention(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return 0


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
            DepthwiseConv1d(second_channels, second_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
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


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(inputs.to(self.device))


class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 10,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)


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





if __name__ == '__main__':
    import numpy as np
    module = ConvolutionModule(in_channels=2)
    data = np.asarray([
        [
            [1, 2],
            [2, 3],
            [2, 3]
        ],
        [
            [1, 2],
            [2, 3],
            [2, 3]
        ],

    ])

    # data = np.asarray([
    #     [[1, 2],
    #      [4, 5],
    #      [4, 5]],
    #
    #     [[1, 2],
    #      [4, 5],
    #      [4, 5]]
    # ])

    output = module(torch.from_numpy(data).float())
    print(output)
