from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
            self,
            audio_paths: list,
            transcripts: list,

    ) -> None:
        super().__init__()


class DataLoader:
    pass