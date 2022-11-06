import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import TrainParams


def train_model(features: pd.DataFrame,
                target: pd.DataFrame,
                params: TrainParams):
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
