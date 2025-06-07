import os
import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.model_selector import get_model
from src.utils.CrisprBERT import base_pair, off_tar_read

# sys.path.remove(str(_parentdir))
# sys.path.insert(0, str(BASE_DIR))


class CrisprDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        config_path = os.path.join("mlflow-serve/mlflow", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.model, _ = get_model(config)

        self.base_p = base_pair()
        self.encoding = "doublet"
        self.base_list = self.base_p.create_dict(self.encoding)

    def predict(self, context, model_input):
        # receive {"sequence": [], "Target sequence":[]}
        # save input data to artifacts/input.csv file with fields "sequence" for RNA and "Target sequence" for DNA

        df = pd.DataFrame(model_input)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/input.csv", index=False)

        encoder = off_tar_read("outputs/input.csv", self.base_list)
        encode_matrix, _ = encoder.encode(self.encoding)

        print(encode_matrix.shape)

        # batch = {"inputs": encode_matrix.astype(np.int64)}
        # self.predict_dataset = CrisprDataset(encode_matrix.astype(np.int64))

        with torch.no_grad():
            inputs = torch.tensor(encode_matrix, dtype=torch.int64).unsqueeze(0)
            outputs = self.model(inputs)
            answer = outputs.numpy()
            print((answer > 0.5).astype(int))
            return (answer > 0.5).astype(int)


@hydra.main(
    version_base=None, config_path=os.path.join(BASE_DIR, "conf/"), config_name="config"
)
def make_model(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with open("mlflow-serve/mlflow/config.yaml", "w") as f:
        yaml.dump(config_dict, f)
    artifacts = {"config": "mlflow-serve/mlflow/config.yaml"}

    mlflow.pyfunc.save_model(
        path="mlflow-serve/mlflow/crispr_off_t_model",
        python_model=CrisprDetector(),
        artifacts=artifacts,
    )


if __name__ == "__main__":
    make_model()
