import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from koasr.models import Conformer


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
            vocab_size=80
            # vocab_size=config.data.vocab_size,
            # input_dim=inputs.size(2),
            # encoder_dim=config.model.encoder_dim,
            # num_encoder_layers=config.model.num_encoder_layers,
            # num_attention_heads=config.model.num_attention_heads,
            # conv_kernel_size=config.model.conv_kernel_size,
            # dropout_p=config.model.dropout_p,
            # max_length=config.train.max_length,
            # device=config.train.device,
        )

    def fit(self):
        temp_audio_data = torch.from_numpy(np.arange(10 * 100 * 80).reshape((10, 100, 80))).float()
        # output = self.model(self.inputs)
        output = self.model(temp_audio_data)
        print(output)


if __name__ == "__main__":
    trainer = Trainer(None, None, None)
    trainer.fit()
