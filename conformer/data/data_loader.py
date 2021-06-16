from torch import Tensor
from torch.utils.data import Dataset
from .audio import load_audio


class AudioDataset(Dataset):
    def __init__(
            self,
            data_paths: list,
            sampling_rate: int = 16000

    ) -> None:
        super().__init__()
        # assert len(audio_paths) == len(transcripts), 'audio_paths and transcripts must be the same length.'
        self.data_paths = data_paths

    def __getitem__(self, index) -> Tensor:
        path, transcript = self.data_paths[index]
        signal = load_audio(path)


class DataLoader:
    pass
