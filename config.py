from dataclasses import dataclass, field

import omegaconf


@dataclass
class Paths:
    raw: str
    processed: str
    interim: str


@dataclass
class Data:
    raw: str
    clean: str
    featured: str
    x_train: str
    x_test: str
    y_train: str
    y_test: str


@dataclass
class Split:
    test_size: float
    random_state: int


@dataclass
class TrainParams:
    model_type: str = field(default='KNeighborsClassifier')
    random_state: int = field(default=3)
    n_neighbors: int = field(default=2)


@dataclass
class Training:
    params: TrainParams
    data: Data


@dataclass
class Cleveland:
    paths: Paths
    data: Data
    split: Split
