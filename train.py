import sys

# dirty
sys.path.append(".")

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from kinodata.data.dataset import KinodataDocked
from kinodata.data.data_module import make_data_module
from kinodata.transform import AddDistancesAndInteractions
from kinodata.model.model import Model


def train(config):
    logger = WandbLogger(log_model="all")

    dataset = KinodataDocked(
        transform=AddDistancesAndInteractions(radius=config.interaction_radius)
    )
    node_types, edge_types = dataset[0].metadata()

    # keyword arguments for the message passing class
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
        use_bonds=config.use_bonds,
        readout_aggregation_type=config.readout_type,
    )

    data_module = make_data_module(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.splitting_seed,
    )

    # logger.watch(model)

    # save model with best validation mean absolute error
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


config_data = dict(
    interaction_radius=5.0,
    splitting_seed=420,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    use_bonds=True,
)

config_model = dict(
    num_mp_layers=4,
    hidden_channels=64,
    act="silu",
    mp_type="rbf",
    mp_reduce="sum",
    rbf_size=64,
    readout_type="sum",
)

config_training = dict(
    lr=3e-4,
    weight_decay=1e-5,
    batch_size=32,
    accumulate_grad_batches=2,
    epochs=100,
    num_workers=32,
    accelerator="gpu",
    loss_type="mse",
)


default_config = config_data | config_model | config_training


if __name__ == "__main__":
    wandb.init(config=default_config, project="kinodata-docked-rescore")
    train(wandb.config)
