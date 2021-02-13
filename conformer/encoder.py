from torch import nn, Tensor
from torch.nn import LayerNorm

from conformer.modules import *


class ConformerBlock(nn.Module):
    """ Conformer Block """
    def __init__(self):
        super().__init__()

        self.ssequential = nn.Sequential(
            ResidualModule(module=FeedForwardModule(), factor=0.5),
            ResidualModule(module=MultiHeadedSelfAttentionModule()),
            ResidualModule(module=ConvolutionModule()),
            ResidualModule(module=FeedForwardModule(), factor=0.5),
            # LayerNorm()
        )


class ConformerEncoder(nn.Module):
    """ Conformer Encoder """
    def __init__(self, num_conformer_block: int = 10):
        super().__init__()
        self.module_list = nn.ModuleList()

        for _ in range(num_conformer_block):
            self.module_list.append(ConformerBlock())

    def forward(self, inputs: Tensor):
        pass
