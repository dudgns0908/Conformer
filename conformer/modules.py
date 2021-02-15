import torch
from torch import nn, Tensor
from .activations import Swish


class MultiHeadedSelfAttentionModule(nn.Module):
    """ Multi-Head Self Attention """

    def __init__(self):
        super().__init__()


class ConvolutionModule(nn.Module):
    """ Convolution Module """

    def __init__(
            self,
            in_channels: int,
            expansion_factor: int = 2,
            kernel_size: int = 32,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device
        inner_channels = in_channels * expansion_factor

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            PointwiseConv1d(in_channels, inner_channels),
            nn.GLU(dim=1),
            DepthwiseConv1d(inner_channels, inner_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(inner_channels),
            Swish(),
            PointwiseConv1d(inner_channels, in_channels),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor):
        inputs = inputs.to(self.device)
        return self.sequential(inputs)


class FeedForwardModule(nn.Module):
    """ Feed Forward Module """

    def __init__(
            self,
            dim: int = 40,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.to(device)

        self.sequential = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(dim, dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs.to(self.device))


class ResidualModule(nn.Module):
    """ Residual Module """

    def __init__(self, module: nn.Module, factor: float = 1.0):
        super().__init__()
        self.module = module
        self.factor = factor

    def forward(self, inputs: Tensor):
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
    ):
        super().__init__()
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.pointwise(x)
        return out


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


if __name__ == '__main__':
    ConvolutionModule()
