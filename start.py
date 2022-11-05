from omegaconf import DictConfig, OmegaConf
import hydra

import src


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def main(cfg: DictConfig):
    src.clean_data(cfg.paths.raw, cfg.paths.clean)
    src.add_features(cfg.paths.clean, cfg.paths.featured)
    src.split_data(cfg.paths.featured, cfg.paths.processed)


if __name__ == "__main__":
    main()
