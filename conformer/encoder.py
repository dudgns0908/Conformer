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
            nn.LayerNorm()
        )


class ConformerEncoder(nn.Module):
    """ Conformer Encoder """

    def __init__(
            self,
            encoder_layers: int = 17,
            encoder_dim: int = 512,
            attention_heads: int = 8,
            conv_kernel_size: int = 32,
            dropout_p: float = 0.1,
            num_conformer_block: int = 10,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        self.spec_augmentation = None  # TODO:: SpecAug 구현
        self.conv_subsampling = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=conv_kernel_size)
        self.liner = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.module_list = nn.ModuleList()
        for _ in range(num_conformer_block):
            self.module_list.append(ConformerBlock())

        self.to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv_subsampling(inputs)
        output = self.liner(output)
        output = self.dropout(output)

        for module in self.module_list:
            output = module(output)

        return output
