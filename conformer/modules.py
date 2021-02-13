import torch
from torch import nn, Tensor


class MultiHeadedSelfAttentionModule(nn.Module):
    """ Multi-Head Self Attention """
    def __init__(self):
        super().__init__()


class ConvolutionModule(nn.Module):
    """ Convolution Module """
    def __init__(self):
        super().__init__()


class FeedForwardModule(nn.Module):
    """ Feed Forward Module """
    def __init__(self):
        super().__init__()


class ResidualModule(nn.Module):
    """ Residual Module """
    def __init__(self, module: nn.Module, factor: float = 1.0):
        super().__init__()
        self.module = module
        self.factor = factor

    def forward(self, inputs: Tensor):
        module_output = self.module(inputs) * self.factor
        return module_output + inputs



