import json
import sys
from typing import NoReturn

import click
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

from config import Cleveland, Training, Running

import src

cs = ConfigStore.instance()
cs.store(name="cleveland_config", node=Cleveland)


@hydra.main(version_base=None, config_path="configs", config_name="prep_conf")
def processing(cfg: Cleveland) -> None:
    src.clean_data(cfg.data.raw, cfg.data.clean)
    src.add_features(cfg.data.clean, cfg.data.featured)
    src.split_data(cfg.data.featured, cfg.paths.processed,
                   cfg.split.test_size, cfg.split.random_state)


@hydra.main(version_base=None, config_path="configs", config_name="train_conf")
def training(cfg: Training) -> None:
    x_train = pd.read_csv(cfg.data.x_train, index_col=0)
    y_train = pd.read_csv(cfg.data.y_train, index_col=0).squeeze()

    model = src.train_model(x_train, y_train, cfg.params)
    file_path = cfg.save_paths.models + cfg.params.model_type + ".sav"
    src.save_model(model, file_path)


@hydra.main(version_base=None, config_path="configs", config_name="train_conf")
def predicting(cfg: Training) -> None:
    x_test = pd.read_csv(cfg.data.x_test, index_col=0)
    y_test = pd.read_csv(cfg.data.y_test, index_col=0).squeeze()
    model_file_path = cfg.save_paths.models + cfg.params.model_type + ".sav"
    model = src.load_model(model_file_path)

    pred = src.predict_model(model, x_test)
    metrics_file_path = cfg.save_paths.metrics + cfg.params.model_type + ".json"
    with open(metrics_file_path, "w") as file:
        json.dump(src.evaluate_model(pred, y_test), file)


def main():
    processing()
    training()
    predicting()


if __name__ == "__main__":
    main()
