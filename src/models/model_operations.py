import pickle
from typing import Union, Dict, NoReturn
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import TrainParams
from src.logs import init_logger

Models = Union[
    LogisticRegression, KNeighborsClassifier,
    GaussianNB, DecisionTreeClassifier,
    SVC, RandomForestClassifier
]

logger_train = init_logger("model_ops_train", "logs/train.log")
logger_predict = init_logger("model_ops_predict", "logs/predict.log")


def train_model(features: pd.DataFrame,
                target: pd.Series,
                params: TrainParams) -> Models:
    if params.model_type == "LogisticRegression" \
            or params.model_type == "LR":
        params.model_type = "LogisticRegression"
        logger_train.info(f"Model type = {params.model_type}")
        model = LogisticRegression()

    elif params.model_type == "KNeighborsClassifier" \
            or params.model_type == "KNN":
        params.model_type = "KNeighborsClassifier"
        logger_train.info(f"Model type = {params.model_type}")
        model = KNeighborsClassifier(n_neighbors=params.n_neighbors)

    elif params.model_type == "GaussianNB" \
            or params.model_type == "GNB":
        params.model_type = "GaussianNB"
        logger_train.info(f"Model type = {params.model_type}")
        model = GaussianNB()

    elif params.model_type == "DecisionTreeClassifier" \
            or params.model_type == "DT":
        params.model_type = "DecisionTreeClassifier"
        logger_train.info(f"Model type = {params.model_type}")
        model = DecisionTreeClassifier()

    elif params.model_type == "SVC":
        model = SVC()

    elif params.model_type == "RandomForestClassifier" \
            or params.model_type == "RF":
        params.model_type = "RandomForestClassifier"
        logger_train.info(f"Model type = {params.model_type}")
        model = RandomForestClassifier()

    else:
        logger_train.error(f"Unknown model type -> {params.model_type}")
        raise NotImplementedError()

    logger_train.info("Model fit.")
    model.fit(features, target)
    return model


def predict_model(model: Models,
                  features: pd.DataFrame) -> np.ndarray:
    logger_predict.info("Model predict.")
    predict = model.predict(features)
    return predict


def evaluate_model(predict: np.ndarray,
                   target: pd.Series) -> Dict[str, float]:
    logger_predict.info("Evaluate some metrics.")
    evaluation = {
        "accuracy_score": accuracy_score(target, predict),
        "f1_score": f1_score(target, predict),
        "r2_score": r2_score(target, predict),
        "mae": mean_absolute_error(target, predict),
        "rmse": mean_squared_error(target, predict, squared=False),
    }
    logger_predict.info(f"Metrics = {evaluation}")
    return evaluation


def save_model(model: Models, save_path: str) -> NoReturn:
    logger_train.info(f"Save model to {save_path}")
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(load_path: str) -> Models:
    logger_predict.info(f"Load model from {load_path}")
    with open(load_path, 'rb') as file:
        model = pickle.load(file)
    return model
