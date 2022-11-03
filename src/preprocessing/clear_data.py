import hydra
from omegaconf import DictConfig
import pandas as pd


@hydra.main(config_path="../../configs", config_name="prep_conf", version_base=None)
def clear_data(conf: DictConfig):
    df = pd.read_csv('../../' + conf.raw_data_path)
    df = df.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
    df.to_csv('../../' + conf.path_to_save)


if __name__ == "__main__":
    clear_data()
