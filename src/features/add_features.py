import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


def add_features(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = df.drop(columns=["Unnamed: 0"])

    a = pd.get_dummies(df["cp"], prefix="cp", drop_first=True)
    b = pd.get_dummies(df["thal"], prefix="thal", drop_first=True)
    c = pd.get_dummies(df["slope"], prefix="slope", drop_first=True)
    d = pd.get_dummies(df["ca"], prefix="ca", drop_first=True)
    e = pd.get_dummies(df["restecg"], prefix="restecg", drop_first=True)

    frames = [df, a, b, c, d, e]
    df = pd.concat(frames, axis=1)

    df = df.drop(columns=["cp", "thal", "slope", "ca", "restecg"])

    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

    X = df[df.columns.difference(['condition'])]
    SKB = SelectKBest(f_classif, k=15).fit(X, np.ravel(df[['condition']]))
    imp_vars_SKB = list(X.columns[SKB.get_support()])

    x = df[imp_vars_SKB]
    y = df['condition']
    df = pd.concat([x, y], axis=1)

    df.to_csv(output_path)

