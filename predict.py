import json

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

from config import Training
import src

logger = src.init_logger("predict_logger", "logs/predict.log")

cs = ConfigStore.instance()
cs.store(name="predict", node=Training)

model_name = {"DT": "DecisionTreeClassifier",
              "GNB": "GaussianNB",
              "KNN": "KNeighborsClassifier",
              "LR": "LogisticRegression",
              "RF": "RandomForestClassifier"}


@hydra.main(version_base=None, config_path="configs", config_name="train_conf")
def predicting(cfg: Training) -> None:
    x_test = pd.read_csv(cfg.data.x_test, index_col=0)
    logger.info(f"Read x_test, it's shape {x_test.shape}")
    y_test = pd.read_csv(cfg.data.y_test, index_col=0).squeeze()
    logger.info(f"Read y_test, it's shape {y_test.shape}")

    for name, full_name in model_name.items():
        if cfg.params.model_type == name:
            cfg.params.model_type = full_name

    model_file_path = cfg.save_paths.models + cfg.params.model_type + ".sav"
    logger.info(f"Load model from {model_file_path}")
    model = src.load_model(model_file_path)

    logger.info("Start predict.")
    pred = src.predict_model(model, x_test)
    logger.info("End predict.")
    metrics_file_path = cfg.save_paths.metrics + cfg.params.model_type + ".json"
    logger.info(f"Save metrics to {metrics_file_path}")
    with open(metrics_file_path, "w") as file:
        json.dump(src.evaluate_model(pred, y_test), file)


if __name__ == "__main__":
    predicting()
