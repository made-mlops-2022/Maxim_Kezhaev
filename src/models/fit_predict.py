from typing import Union, Dict

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

Models = Union[
    LogisticRegression, KNeighborsClassifier,
    GaussianNB, DecisionTreeClassifier,
    SVC, RandomForestClassifier
]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                params: TrainParams) -> Models:
    if params.model_type == "LogisticRegression":
        model = LogisticRegression()
    elif params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(n_neighbors=params.n_neighbors)
    elif params.model_type == "GaussianNB":
        model = GaussianNB()
    elif params.model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier()
    elif params.model_type == "SVC":
        model = SVC()
    elif params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_model(model: Models,
                  features: pd.DataFrame) -> np.ndarray:
    predict = model.predict(features)
    return predict


def evaluate_model(predict: np.ndarray,
                   target: pd.Series) -> Dict[str, float]:
    evaluation = {
        "accuracy_score": accuracy_score(target, predict),
        "f1_score": f1_score(target, predict),
        "r2_score": r2_score(target, predict),
        "mae": mean_absolute_error(target, predict),
        "rmse": mean_squared_error(target, predict, squared=False),
    }
    return evaluation
