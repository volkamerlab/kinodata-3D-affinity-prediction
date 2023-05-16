import sys

# dirty
sys.path.append(".")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

from kinodata.model.dti_baseline import (
    DTIModel,
    LigandGINE,
    KissimPocketTransformer,
    GlobalSumDecoder,
)
import kinodata.configuration as configuration
from train_regressor import make_data
from kinodata.data.dataset import KinodataDocked

from pathlib import Path
import sys
from typing import Callable, Optional, Tuple, Dict

# dirty
sys.path.append(".")

from functools import partial

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.transforms import Compose
import pandas as pd

from kinodata.data.dataset import KinodataDocked
from kinodata.data.artifical_decoys import KinodataDockedWithDecoys
from kinodata.data.data_module import make_data_module
from kinodata.data.data_split import Split
from kinodata.transform import AddDistancesAndInteractions, PerturbAtomPositions
from kinodata.transform.add_global_attr_to_edge import AddGlobalAttrToEdge
from kinodata.model.regression_model import RegressionModel
from kinodata.model.egnn import EGNN
from kinodata.model.egin import HeteroEGIN
from kinodata.types import EdgeType

from kinodata import configuration


def main(config):
    logger = WandbLogger(log_model="all")
    data_module = make_data(config)
    model = DTIModel(config, LigandGINE, KissimPocketTransformer, GlobalSumDecoder)
    val_checkpoint_callback = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
    )
    lr_monitor = LearningRateMonitor("epoch")

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[val_checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":

    import torch

    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    configuration.register(
        "dti_baseline",
        num_layers=3,
        hidden_channels=128,
        act="silu",
        num_attention_blocks=2,
        kissim_size=12,
    )

    config = configuration.get("data", "training", "dti_baseline")
    config = config.update_from_args()
    config["num_workers"] = 32

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["dti"])
    main(config)
