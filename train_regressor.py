import sys
from typing import Tuple

# dirty
sys.path.append(".")

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.transforms import Compose

from kinodata.data.dataset import KinodataDocked
from kinodata.data.artifical_decoys import KinodataDockedWithDecoys
from kinodata.data.data_module import make_data_module
from kinodata.data.data_split import RandomSplit
from kinodata.transform import AddDistancesAndInteractions
from kinodata.model.regression_model import RegressionModel
from kinodata import configuration


def make_model(node_types, edge_types, config) -> RegressionModel:

    # keyword arguments for the message passing class
    mp_kwargs = {
        "rbf_size": config.rbf_size,
        "interaction_radius": config.interaction_radius,
        "reduce": config.mp_reduce,
    }

    if config.checkpoint is not None:
        ...

    model = RegressionModel(
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
        use_bonds=config.use_bonds,
        readout_aggregation_type=config.readout_type,
    )

    return model


def make_data(config) -> Tuple[KinodataDocked, LightningDataset]:

    transforms = [AddDistancesAndInteractions(radius=config.interaction_radius)]

    if config.add_artificial_decoys:
        dataset = KinodataDockedWithDecoys(transform=Compose(transforms))
    else:
        dataset = KinodataDocked(transform=Compose(transforms))

    if config.cold_split:
        raise NotImplementedError()
    else:
        split = RandomSplit(config.train_size, config.val_size, config.test_size)

    data_module = make_data_module(
        dataset=dataset,
        splitting_method=split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return dataset, data_module


def train_regressor(config):
    logger = WandbLogger(log_model="all")

    dataset, data_module = make_data(config)
    node_types, edge_types = dataset[0].metadata()
    model = make_model(node_types, edge_types, config)
    val_checkpoint_callback = ModelCheckpoint(monitor="val_mae", mode="min")

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[val_checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    config = configuration.get("data", "model", "training")
    config = configuration.overwrite_from_file(config, "config_regressor_local.yaml")
    print(config)
    wandb.init(config=config, project="kinodata-docked-rescore")
    train_regressor(wandb.config)
