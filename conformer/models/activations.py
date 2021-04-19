import torch
from torch import nn, Tensor


class Swish(nn.Module):
    """ Swish Activation Function """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor):
        return self.sigmoid(inputs) * inputs
