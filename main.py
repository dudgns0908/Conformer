import logging
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf

#
# @dataclass
# class TestConfig:
#     phase: str = 'train'


log = logging.getLogger(__name__)


@hydra.main(config_path='./config', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg)


if __name__ == '__main__':
    main()
