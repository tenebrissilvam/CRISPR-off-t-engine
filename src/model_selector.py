from typing import Any, Dict

from data_modules.CRISPR_DIPOFF import RNNDataModule
from data_modules.CrisprBERT import CrisprDataModule
from data_modules.DeepCNN import CNNDataModule
from data_modules.R_CRISPR import RCRISPRDataModule
from models.CRISPR_DIPOFF import RNNLightningModel
from models.CrisprBERT import CrisprBERTLightning
from models.DeepCNN import CNNModel
from models.R_CRISPR import CrisprNetLightning


def get_model(model_conf: Dict[str, Any]):
    """Select model to run experiments"""
    label = model_conf.model.label
    if label == "crispr_bert":
        model = CrisprBERTLightning(
            **model_conf.model.crispr_bert,
        )
        data_module = CrisprDataModule(
            **model_conf.data_module.crispr_bert,
        )
        return model, data_module
    if label == "crispr_dipoff":
        model = RNNLightningModel(**model_conf.model.crispr_dipoff)
        data_module = RNNDataModule(**model_conf.data_module.crispr_dipoff)
        return model, data_module
    if label == "r_crispr":
        model = CrisprNetLightning(**model_conf.model.r_crispr)
        data_module = RCRISPRDataModule(**model_conf.data_module.r_crispr)
        return model, data_module

    if label == "deep_cnn":
        model = CNNModel(**model_conf.model.deep_cnn)
        data_module = CNNDataModule(**model_conf.data_module.deep_cnn)
        return model, data_module
    else:
        raise ValueError(f"There is no such model with label {label}")
