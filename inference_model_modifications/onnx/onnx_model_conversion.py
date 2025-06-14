import os
import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
from src.model_selector import get_model
from src.utils.CrisprBERT import base_pair, off_tar_read


@hydra.main(
    version_base=None,
    config_path=os.path.join(BASE_DIR, "conf/"),
    config_name="config",
)
def export_onnx(cfg):
    """
    export_onnx

    convert pytorch lightning model to onnx format

    :param cfg: hydra config
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    model, _ = get_model(config_dict)
    model.eval()
    model.to(device)

    base_p = base_pair()
    encoding = "doublet"
    base_list = base_p.create_dict(encoding)

    model_input = {
        "sequence": ["GTCACCAATCCTGTCCCTAGTGG"],
        "Target sequence": ["TAAAGCAATCCTGTCCCCAGAGT"],
    }

    df = pd.DataFrame(model_input)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/input.csv", index=False)

    encoder = off_tar_read("outputs/input.csv", base_list)
    encode_matrix, _ = encoder.encode(encoding)

    dummy_input = torch.tensor(encode_matrix, dtype=torch.int32).unsqueeze(0)
    dummy_input = dummy_input.to(device)

    onnx_path = Path("inference_model_modifications/onnx") / "crispr_detector.onnx"
    os.makedirs(onnx_path.parent, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model successfully exported to {onnx_path.resolve()}")


if __name__ == "__main__":
    export_onnx()
