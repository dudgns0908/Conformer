from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    epoch: int = 100
    max_length: int = 80

    device: str = 'cpu'
