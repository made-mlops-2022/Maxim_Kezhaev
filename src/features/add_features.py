import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from src.logs import init_logger

logger = init_logger('features_logger', 'logs/train.log')


def add_features(input_path: str, output_path: str):
    df = pd.read_csv(input_path, index_col=0)
    logger.info("Read dataset.")

    a = pd.get_dummies(df["cp"], prefix="cp", drop_first=True)
    b = pd.get_dummies(df["thal"], prefix="thal", drop_first=True)
    c = pd.get_dummies(df["slope"], prefix="slope", drop_first=True)
    d = pd.get_dummies(df["ca"], prefix="ca", drop_first=True)
    e = pd.get_dummies(df["restecg"], prefix="restecg", drop_first=True)

    frames = [df, a, b, c, d, e]
    df = pd.concat(frames, axis=1)
    logger.info("Concat dataset.")

    df = df.drop(columns=["cp", "thal", "slope", "ca", "restecg"])
    logger.info("Delete extra columns.")

    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
    logger.info("Add Standard Scaler.")

    X = df[df.columns.difference(['condition'])]
    SKB = SelectKBest(f_classif, k=13).fit(X, np.ravel(df[['condition']]))
    imp_vars_SKB = list(X.columns[SKB.get_support()])
    logger.info(f"Select top-columns. {imp_vars_SKB}")

    x = df[imp_vars_SKB]
    y = df['condition']
    df = pd.concat([x, y], axis=1)
    logger.info("Concat with targer column.")

    df.to_csv(output_path)
    logger.info(f"Save changed dataset to {output_path}")
