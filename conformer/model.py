import torch
from torch import nn, Tensor
from torch.optim import Adam

from conformer.encoder import ConformerEncoder


class Conformer(nn.Module):
    """ Conformer Model """

    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            attention_heads: int = 8,
            conv_kernel_size: int = 31,
            dropout_p: float = 0.1,
            decoder_dim: int = 640,
            num_decoder_layers: int = 1,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.device = device

        # Encoder
        self.encoder_dim = encoder_dim
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            attention_heads=attention_heads,
            conv_kernel_size=conv_kernel_size,
            dropout_p=dropout_p,
        ).to(self.device)

        # Decoder
        self.decoder = None

    def forward(self, inputs: Tensor) -> Tensor:
        encoder_val = self.encoder(inputs)
        return Tensor([])

    def fit(self, inputs: Tensor, labels: Tensor) -> None:
        lr = 0.05 / (self.encoder_dim ** 0.5)
        optimizer = Adam(self.encoder.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-10)


