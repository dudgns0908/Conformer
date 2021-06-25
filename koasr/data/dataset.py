import os
from typing import Tuple, Any, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from .audio import load_audio


class AudioDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            audio_paths: list,
            transcripts: list,
            sampling_rate: int = 16000,
            sos_id: int = 1,
            eos_id: int = 2,

    ) -> None:
        super().__init__()
        assert len(audio_paths) == len(transcripts), 'audio_paths and transcripts must be the same length.'

        self.dataset_path = dataset_path
        self.audio_paths = audio_paths
        self.transcripts = transcripts

    def __getitem__(self, index) -> Tuple[Tensor, Union[list, None]]:
        data_path = os.path.join(self.dataset_path, self.audio_paths[index])
        data = self._get_audio_feature(data_path)
        transcript = None if self.transcripts is None else self.transcripts[index]

        return data, transcript

    def _get_audio_feature(self, path: str, augment):
        signal = load_audio(path)
        return signal
        # feature = self.transforms(signal)
        # feature -= feature.mean()
        # feature /= np.std(feature)
        # feature = torch.FloatTensor(feature).transpose(0, 1)

        # if augment == self.SPEC_AUGMENT:
        #     feature = self._spec_augment(feature)

        # return feature
