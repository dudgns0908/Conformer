import os
from typing import Tuple, Any, Union

from torch import Tensor
from torch.utils.data import Dataset
from .audio import load_audio


class AudioDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            data_paths: list,
            transcripts: list = None,
            sampling_rate: int = 16000,
            sos_id: int = 1,
            eos_id: int = 2,

    ) -> None:
        super().__init__()
        assert len(data_paths) == len(transcripts), 'audio_paths and transcripts must be the same length.'

        self.dataset_path = dataset_path
        self.data_paths = data_paths
        self.transcripts = transcripts

    def __getitem__(self, index) -> Tuple[Tensor, Union[list, None]]:
        data_path = os.path.join(self.dataset_path, self.data_paths[index])
        data = self._get_audio_feature(data_path)

        transcript = None
        if self.transcripts is not None:
            transcript = self.transcripts[index]

        return data, transcript

    def _get_audio_feature(self, path: str, ):
        signal = load_audio(path)
        return signal




class DataLoader:
    pass
