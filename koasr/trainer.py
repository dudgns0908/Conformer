import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from koasr.data.data_loader import AudioDataset
from koasr.models import Conformer
from koasr.utils.model_info import model_dict


class Trainer:
    def __init__(
            self,
            dataset_dir: str,
            dataset_name: str,
            model_name: str,
            model_params: dict,
    ):
        assert model_name in model_dict.keys(), f'This is Not supported model name ({model_name})'

        self.dataset_dir = dataset_dir
        self.transcript_path = transcript_path


        self.datasets = AudioDataset(dataset_dir, dataset_name)
        # self.model = model_dict[model_name](vocab_size=80, **model_params)
        self.model = Conformer(**model_params)

    def fit(self):
        temp_audio_data = torch.from_numpy(np.arange(10 * 100 * 80).reshape((10, 100, 80))).float()
        # output = self.model(self.inputs)
        output = self.model(temp_audio_data)
        print(output)


if __name__ == "__main__":
    trainer = Trainer(None, None, None)
    trainer.fit()
