import hydra
from hydra.core.config_store import ConfigStore

from config import Cleveland
import src


cs = ConfigStore.instance()
cs.store(name="cleveland_config", node=Cleveland)


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def main(cfg: Cleveland):
    src.clean_data(cfg.data.raw, cfg.data.clean)
    src.add_features(cfg.data.clean, cfg.data.featured)
    src.split_data(cfg.data.featured, cfg.paths.processed)


if __name__ == "__main__":
    main()
