import torch
from torch import nn, Tensor
from torch.nn.modules import activation


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
