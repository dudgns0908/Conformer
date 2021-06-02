import logging
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

#
# @dataclass
# class TestConfig:
#     phase: str = 'train'
from conformer.trainer import Trainer

log = logging.getLogger(__name__)


def train(config: DictConfig):
    # Load data
    inputs = torch.from_numpy(np.arange(10 * 100 * 80).reshape((10, 100, 80))).float()
    transcripts = [[1, 2, 3, 4, 5, 56]]

    # get model
    trainer = Trainer(config)

    # train
    trainer.fit(
        inputs=inputs,
        transcripts=transcripts,
        num_epoch=config.num_epoch
    )


@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg)


if __name__ == '__main__':
    main()
