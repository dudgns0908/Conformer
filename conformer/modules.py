import torch
from torch import nn, Tensor
from conformer.activations import Swish
from conformer.layers import Transpose, PointwiseConv1d, DepthwiseConv1d


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

        # nn.Embedding.from_pretrained(self.get_sinusoid_encoding_table(src_len+1 , d_model), freeze=True)

        self.P = 2 ** 12
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, num_heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, num_heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((num_heads, self.d_k)), requires_grad=True)

        self.sequential = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttentionWithRelativePositionalEmbedding(),
            nn.Dropout(p=dropout_p)
        )

    def get_sinusoid_encoding_table(self, n_position, d_model):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs.to(self.device))


class MultiHeadAttentionWithRelativePositionalEmbedding(nn.Module):
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
            MultiHeadAttentionWithRelativePositionalEmbedding(),
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
