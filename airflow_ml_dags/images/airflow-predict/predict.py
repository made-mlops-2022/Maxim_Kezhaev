import os
import pandas as pd
import pickle
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-path")
@click.option("--output-dir")
def predict(input_dir: str, model_path: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(data)
    pred = pd.DataFrame(pred)
    os.makedirs(output_dir, exist_ok=True)
    pred.to_csv(os.path.join(output_dir, "prediction.csv"), index=False)


if __name__ == '__main__':
    predict()
