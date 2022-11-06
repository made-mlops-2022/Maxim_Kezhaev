import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.metrics import accuracy_score

from config import Cleveland, Training

import src

cs = ConfigStore.instance()
cs.store(name="cleveland_config", node=Cleveland)


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def processing(cfg: Cleveland):
    src.clean_data(cfg.data.raw, cfg.data.clean)
    src.add_features(cfg.data.clean, cfg.data.featured)
    src.split_data(cfg.data.featured, cfg.paths.processed,
                   cfg.split.test_size, cfg.split.random_state)


@hydra.main(version_base=None, config_path="configs", config_name="train_conf")
def training(cfg: Training):

    x_train = pd.read_csv(cfg.data.x_train, index_col=0)
    y_train = pd.read_csv(cfg.data.y_train, index_col=0).squeeze()

    x_test = pd.read_csv(cfg.data.x_test, index_col=0)
    y_test = pd.read_csv(cfg.data.y_test, index_col=0).squeeze()

    model = src.train_model(x_train, y_train, cfg.params)
    pred = src.predict_model(model, x_test)
    print(src.evaluate_model(pred, y_test))


if __name__ == "__main__":
    training()
