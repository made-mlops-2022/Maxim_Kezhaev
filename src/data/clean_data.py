import pandas as pd


def clean_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = df.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
    df.to_csv(output_path)

