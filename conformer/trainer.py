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
    temp_audio_data = torch.from_numpy(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 20]]))

    trainer = Trainer()
    trainer.fit(temp_audio_data, None)
