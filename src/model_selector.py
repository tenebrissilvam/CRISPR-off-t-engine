import sys
from pathlib import Path
from typing import Any, Dict

import torch

BASE_DIR = Path(__file__).resolve().parents[1]  # 2
sys.path.insert(0, str(BASE_DIR))

from src.data_modules.CRISPR_DIPOFF import RNNDataModule
from src.data_modules.CrisprBERT import CrisprDataModule
from src.data_modules.DeepCNN import CNNDataModule
from src.data_modules.R_CRISPR import RCRISPRDataModule
from src.models.CRISPR_DIPOFF import RNNLightningModel
from src.models.CrisprBERT import CrisprBERTLightning
from src.models.DeepCNN import CNNModel
from src.models.R_CRISPR import CrisprNetLightning


def get_model(model_conf: Dict[str, Any]):
    """Select model to run experiments"""
    mode = model_conf["mode"]["label"]

    label = model_conf["model"]["label"]
    if label == "crispr_bert":
        data_module = CrisprDataModule(
            **model_conf["data_module"]["crispr_bert"],
        )
        if mode == "train":
            torch.backends.cudnn.enabled = True
            model = CrisprBERTLightning(**model_conf.model.crispr_bert)
        else:
            torch.backends.cudnn.enabled = False
            model = CrisprBERTLightning.load_from_checkpoint(
                model_conf["mode"]["checkpoint"],
            ).eval()
        return model, data_module
    if label == "crispr_dipoff":
        data_module = RNNDataModule(**model_conf.data_module.crispr_dipoff)
        if mode == "train":
            model = RNNLightningModel(**model_conf.model.crispr_dipoff)
        else:
            model = RNNLightningModel.load_from_checkpoint(
                model_conf.mode.checkpoint,
            ).eval()
        return model, data_module
    if label == "r_crispr":
        data_module = RCRISPRDataModule(**model_conf.data_module.r_crispr)
        if mode == "train":
            model = CrisprNetLightning(**model_conf.model.r_crispr)
        else:
            model = CrisprNetLightning.load_from_checkpoint(
                model_conf.mode.checkpoint,
            ).eval()
        return model, data_module

    if label == "deep_cnn":
        data_module = CNNDataModule(**model_conf.data_module.deep_cnn)
        if mode == "train":
            model = CNNModel(**model_conf.model.deep_cnn)

        else:
            model = CNNModel.load_from_checkpoint(
                model_conf.mode.checkpoint,
            ).eval()
        return model, data_module
    else:
        raise ValueError(f"There is no such model with label {label}")
