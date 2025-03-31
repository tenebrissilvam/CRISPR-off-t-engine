from typing import Any, Dict

import pytorch_lightning as pl


def get_logger(logging_conf: Dict[str, Any]) -> pl.loggers.logger.Logger:
    """Select logger to log experiments"""
    label = logging_conf["label"]
    if label == "wandb":
        return pl.loggers.WandbLogger(
            project=logging_conf["project"],
            name=logging_conf["name"],
            save_dir=logging_conf["save_dir"],
        )
    else:
        raise ValueError(f"There is no such logger with label {label}")
