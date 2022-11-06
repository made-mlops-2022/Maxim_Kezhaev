from dataclasses import dataclass


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
class Cleveland:
    paths: Paths
    data: Data
