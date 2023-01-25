from pathlib import Path
import sys
import os

sys.path.append(".")

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from kinodata.data.dataset import KinodataDocked
from kinodata.data.data_module import make_data_module
from kinodata.transform import AddDistancesAndInteractions
from kinodata.model.model import Model

import yaml


def train(config):
    logger = WandbLogger(log_model=True)

    dataset = KinodataDocked(
        transform=AddDistancesAndInteractions(radius=config.interaction_radius)
    )
    node_types, edge_types = dataset[0].metadata()
    mp_kwargs = {
        "rbf_size": config.rbf_size,
        "interaction_radius": config.rbf_size,
        "reduce": config.mp_reduce,
    }
    model = Model(
        node_types=node_types,
        edge_types=edge_types,
        hidden_channels=config.hidden_channels,
        num_mp_layers=config.num_mp_layers,
        act=config.act,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        mp_type=config.mp_type,
        loss_type=config.loss_type,
        mp_kwargs=mp_kwargs,
    )

    data_module = make_data_module(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        seed=420,
        log_seed=True,
    )

    # logger.watch(model)

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=data_module)


default_config = dict(
    batch_size=16,
    accumulate_grad_batches=4,
    num_mp_layers=4,
    hidden_channels=4,
    lr=3e-4,
    act="silu",
    weight_decay=1e-5,
    interaction_radius=5.0,
    epochs=100,
    num_workers=16,
    mp_type="rbf",
    mp_reduce="sum",
    rbf_size=64,
    accelerator="cpu",
    loss_type="mse",
)


if __name__ == "__main__":
    wandb.init(config=default_config, project="kinodata-docked-rescore")
    train(wandb.config)
