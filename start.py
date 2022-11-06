from omegaconf import DictConfig, OmegaConf
import hydra

import src


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def main(cfg: DictConfig):
    src.clean_data(cfg.data.raw, cfg.data.clean)
    src.add_features(cfg.data.clean, cfg.data.featured)
    src.split_data(cfg.data.featured, cfg.paths.processed)


if __name__ == "__main__":
    main()
