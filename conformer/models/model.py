__all__ = ['Conformer']

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
from torch import Tensor
from torch.optim import Adam

from conformer.models.decoder import ConformerDecoder
from conformer.models.encoder import ConformerEncoder


class Conformer(pl.LightningModule):
    """ Conformer Model """

    def __init__(
            self,
            vocab_size: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            conv_kernel_size: int = 31,
            dropout_p: float = 0.1,

            # Decoder
            decoder_name: str = None,
            decoder_dim: int = 640,
            num_decoder_layers: int = 1,
            max_length: int = 300,

            # Device
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
            num_attention_heads=num_attention_heads,
            conv_kernel_size=conv_kernel_size,
            dropout_p=dropout_p,
        ).to(self.device)

        # Decoder
        self.decoder = ConformerDecoder(
            vocab_size=vocab_size,
            hidden_size=encoder_dim,
            max_length=max_length,
            device=self.device,
        ).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        encoder_output = self.encoder(inputs)
        output = self.decoder(encoder_output)
        return output

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        loss = torch.nn.CTCLoss()

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def configure_optimizers(self):
        lr = 0.05 / (self.encoder_dim ** 0.5)
        optimizer = Adam(self.encoder.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-10)
        return optimizer

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass



