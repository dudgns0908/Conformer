import numpy as np
import torch
from torch import Tensor

from conformer.models import Conformer


class Trainer:
    def __init__(self):
        self.model = Conformer()

    def fit(self, inputs: Tensor, y: Tensor):
        output = self.model(inputs)
        print(output)


if __name__ == "__main__":
    temp_audio_data = torch.from_numpy(np.arange(2 * 257 * 80).reshape((2, 257, 80))).float()

    trainer = Trainer()
    trainer.fit(temp_audio_data, [])
