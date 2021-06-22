from torch import Tensor
from torch.utils.data import Dataset
from .audio import load_audio


class AudioDataset(Dataset):
    def __init__(
            self,
            data_paths: list,
            transcripts: list,
            sampling_rate: int = 16000,
            sos_id: int = 1,
            eos_id: int = 2,

    ) -> None:
        super().__init__()
        assert len(data_paths) == len(transcripts), 'audio_paths and transcripts must be the same length.'
        self.data_paths = data_paths

    def __getitem__(self, index) -> Tensor:
        path, transcript = self.data_paths[index]
        signal = load_audio(path)


class DataLoader:
    pass
