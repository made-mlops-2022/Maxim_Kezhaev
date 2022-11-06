import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_path: str,
               output_path: str,
               test_size: float,
               random_state: int):
    df = pd.read_csv(input_path, index_col=0)

    X = df[df.columns.difference(['condition'])]
    y = df['condition']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    x_train.to_csv(output_path + "/X_train.csv")
    x_test.to_csv(output_path + "/X_test.csv")
    y_train.to_csv(output_path + "/y_train.csv")
    y_test.to_csv(output_path + "/y_test.csv")
