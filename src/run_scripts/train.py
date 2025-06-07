import os
import pathlib
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl

BASE_DIR = Path(__file__).resolve().parents[2]

_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(BASE_DIR)

from logger_selector import get_logger
from model_selector import get_model

sys.path.remove(str(_parentdir))
sys.path.insert(0, str(BASE_DIR))


@hydra.main(
    version_base=None, config_path=os.path.join(BASE_DIR, "conf/"), config_name="config"
)
def run(cfg):
    model, data_module = get_model(cfg)
    logger = get_logger(cfg.logging)

    print("Running in train mode")

    trainer = pl.Trainer(
        max_epochs=cfg.experiments.max_epochs,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                **cfg.experiments.train.callbacks.model_checkpoint
            ),
            pl.callbacks.LearningRateMonitor(
                **cfg.experiments.train.callbacks.lr_rate_monitor
            ),
        ],
    )

    trainer.fit(model, datamodule=data_module)

    print("Training finished")


if __name__ == "__main__":
    run()
