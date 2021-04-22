import torch
from torch import nn, Tensor
from torch.nn import LayerNorm

from conformer.models.modules import *
from conformer.models.convolutions import Conv2dSubsampling



class ConformerBlock(nn.Module):
    """ Conformer Block """

    def __init__(
            self,
            dim: int = 512,
            num_attention_heads: int = 8,
            conv_expansion_factor: int = 2,
            conv_kernel_size: int = 32,
    ):
        super().__init__()

        self.ssequential = nn.Sequential(
            ResidualModule(module=FeedForwardModule(dim), factor=0.5),

            # TODO:: 2. Multi Head Attention
            ResidualModule(module=MultiHeadedSelfAttentionModule(
                dim=dim,
                num_heads=num_attention_heads,
            )),

            ResidualModule(module=ConvolutionModule(
                in_channels=dim,
                expansion_factor=conv_expansion_factor,
                kernel_size=conv_kernel_size
            )),

            ResidualModule(module=FeedForwardModule(dim), factor=0.5),
            nn.LayerNorm(dim)
        )


class ConformerEncoder(nn.Module):
    """ Conformer Encoder """

    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            conv_expansion_factor: int = 2,
            conv_kernel_size: int = 32,
            dropout_p: float = 0.1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        # This is an augmentation and is not necessarily implemented.
        self.spec_augmentation = None

        # TODO:: 1. complete subsampling
        self.conv_subsampling = Conv2dSubsampling(1, encoder_dim)
        self.liner = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        # self.module_list = nn.ModuleList([ConformerBlock() for _ in range(encoder_layers)])
        self.conformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conformer_block = ConformerBlock(
                dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
            )
            self.conformer_blocks.append(conformer_block)

        self.to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv_subsampling(inputs)
        output = self.liner(output)
        output = self.dropout(output)

        for conformer_block in self.conformer_blocks:
            output = conformer_block(output)

        return output
