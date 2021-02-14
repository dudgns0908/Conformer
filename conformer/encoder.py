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

    def __init__(
            self,
            encoder_dim: int = 512,
            encoder_num_layers: int = 17,
            attention_head: int = 8,
            conv_kernel_size: int = 32,
            dropout_p: float = 0.1,
            num_conformer_block: int = 10
    ):
        super().__init__()

        self.spec_augmentation = None  # TODO:: SpecAug 구현
        self.conv_subsampling = nn.Conv1d(encoder_dim, encoder_dim,
                                          kernel_size=conv_kernel_size)  # TODO:: Convolution subsampling 구현
        self.liner = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.module_list = nn.ModuleList()
        for _ in range(num_conformer_block):
            self.module_list.append(ConformerBlock())

    def forward(self, inputs: Tensor):
        pass
