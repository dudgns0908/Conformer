import hydra
from omegaconf import DictConfig, OmegaConf
from conformer.utils import logger


@hydra.main(config_path='./config', config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
