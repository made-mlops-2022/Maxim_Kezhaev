import json

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

from config import Training

import src

cs = ConfigStore.instance()
cs.store(name="predict", node=Training)


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


if __name__ == "__main__":
    predicting()
