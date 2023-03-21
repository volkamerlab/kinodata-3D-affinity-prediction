import sys

# dirty
sys.path.append(".")

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch_geometric.nn.models import GIN
from torch_geometric.nn.resolver import aggregation_resolver

from kinodata.model.ligand_gin_baseline import LigandGNNBaseline
from kinodata import configuration

from train_regressor import make_data


def make_model(config) -> LigandGNNBaseline:
    if config.gnn_type == "gin":
        encoder = GIN(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=config.hidden_channels,
            act=config.act,
        )
    else:
        raise ValueError(config.gnn_type)

    readout = aggregation_resolver(config.readout_type)

    model = LigandGNNBaseline(encoder, readout)
    return model


def train_baseline(config):
    logger = WandbLogger(log_model="all")

    data_module = make_data(config)
    model = make_model(config)
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
    configuration.register(
        "ligand_gnn_baseline",
        gnn_type="gin",
        hidden_channels=64,
        num_layers=4,
        act="silu",
        readout_type="sum",
    )
    config = configuration.get("data", "training", "ligand_gnn_baseline")
    config = configuration.overwrite_from_file(
        config, "config_ligand_baseline_local.yaml"
    )
    config = config.update_from_args()

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["ligand-only"])

    wandb.agent()
    train_baseline(wandb.config)
