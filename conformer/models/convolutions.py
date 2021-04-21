from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor


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
            kernels_per_layer: int = 1,
            kernel_size: int = 10,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels * kernels_per_layer, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)


class Conv2dSubsampling(nn.Module):
    """ Conv2dSubsampling """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: nn.Module = nn.ReLU
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 2),
            activation(),
            torch.nn.Conv2d(out_channels, out_channels, 3, 2),
            activation()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv(inputs.unsqueeze(1).transpose(2, 3))

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2).contiguous().view(batch_size, seq_lengths, channels * dimension)
        return outputs
