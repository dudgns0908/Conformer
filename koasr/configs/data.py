from dataclasses import dataclass


@dataclass
class DataConfig:
    audio_path: str = '../dataset/*.pcm'
    dataset_download: bool = True
    vocab_size: int = 5000