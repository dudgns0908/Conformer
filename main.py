import logging
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

#
# @dataclass
# class TestConfig:
#     phase: str = 'train'
from conformer.configs.data import DataConfig
from conformer.configs.model import ConformerLargeConfig
from conformer.configs.train import TrainConfig
from conformer.trainer import Trainer

log = logging.getLogger(__name__)


def train(config: DictConfig):
    # Load data
    batch_size = 10

    input_dim = 80
    inputs = torch.from_numpy(np.arange(batch_size * 100 * input_dim).reshape((batch_size, 100, input_dim))).float()
    transcripts = torch.from_numpy([[1, 2, 3, 4, 5, 56]])

    # get model
    trainer = Trainer(
        config=config,
        inputs=inputs,
        transcripts=transcripts,
    )

    # train
    trainer.fit()


cs = ConfigStore.instance()
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="model", name="conformer-large", node=ConformerLargeConfig)


@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # train(cfg)


if __name__ == '__main__':
    main()
