from dataclasses import dataclass, field


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
class Cleveland:
    paths: Paths
    data: Data
    split: Split


@dataclass
class TrainParams:
    model_type: str
    random_state: int
    n_neighbors: int = field(default=2)

