from omegaconf import DictConfig, OmegaConf
import hydra

import src


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def main(cfg: DictConfig):
    src.clean_data(cfg.raw_data_path, cfg.clean_data_path)
    src.add_features(cfg.clean_data_path, cfg.featured_data_path)
    src.split_data(cfg.featured_data_path, cfg.path_to_save)


if __name__ == "__main__":
    main()
