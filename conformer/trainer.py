import numpy as np
import torch
from torch import Tensor

from conformer.models import Conformer


class Trainer:
    def __init__(self):
        self.model = Conformer()

    def fit(self, x: Tensor, y: Tensor):
        output = self.model(temp_audio_data)
        print(output)


if __name__ == "__main__":
    temp_audio_data = torch.from_numpy(np.arange(257 * 80).reshape((1, 257, 80))).float()

    trainer = Trainer()
    trainer.fit(temp_audio_data, None)
