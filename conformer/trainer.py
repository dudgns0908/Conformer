import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from conformer.data.dataset import AudioDataset
from conformer.models.model import Conformer
from conformer.utils.model_info import model_dict


class Trainer:
    def __init__(
            self,
            dataset_dir: str,
            model_name: str,
            model_params: dict,
            resume: bool = False,
    ):
        assert model_name.lower() in model_dict.keys(), f'This is Not supported model name ({model_name})'

        self.dataset_dir = dataset_dir
        # self.transcript_path = transcript_path

        # Read data and preprocessing
        self.datasets = AudioDataset(dataset_dir, audio_paths=[], transcripts=[])
        # self.model = model_dict[model_name](vocab_size=80, **model_params)

        # Load model
        self.model = Conformer(**model_params)

    def fit(self):
        temp_audio_data = torch.from_numpy(np.arange(10 * 100 * 80).reshape((10, 100, 80))).float()
        # output = self.model(self.inputs)
        output = self.model(temp_audio_data)
        print(output)


if __name__ == "__main__":
    model_name = 'Conformer'
    trainer = Trainer(None, model_name, {'vocab_size': 80})
    trainer.fit()
