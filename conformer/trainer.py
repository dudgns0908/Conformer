import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from conformer.models import Conformer


class Trainer:
    def __init__(
            self,
            config: DictConfig,
            inputs: Tensor,
            transcripts: Tensor,
    ):
        self.config = config
        self.inputs = inputs
        self.transcripts = transcripts

        self.model = Conformer(
            vocab_size=config.vocab_size,
            input_dim=inputs.shape(1),
            encoder_dim=config.encoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_attention_heads=config.num_attention_heads,
            conv_kernel_size=config.conv_kernel_size,
            dropout_p=config.dropout_p,
            max_length=config.max_length,
            device=config.device,
        )

    def fit(self):
        output = self.model(self.inputs)
        print(output)


if __name__ == "__main__":
    temp_audio_data = torch.from_numpy(np.arange(10 * 100 * 80).reshape((10, 100, 80))).float()

    trainer = Trainer()
    trainer.fit(temp_audio_data, [])
