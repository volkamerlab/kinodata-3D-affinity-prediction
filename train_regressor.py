import sys
from typing import Callable, Tuple

# dirty
sys.path.append(".")

from functools import partial

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.transforms import Compose

from kinodata.data.dataset import KinodataDocked
from kinodata.data.artifical_decoys import KinodataDockedWithDecoys
from kinodata.data.data_module import make_data_module
from kinodata.data.data_split import RandomSplit
from kinodata.transform import AddDistancesAndInteractions, PerturbAtomPositions
from kinodata.transform.add_global_attr_to_edge import AddGlobalAttrToEdge
from kinodata.model.regression_model import RegressionModel
from kinodata.model.egnn import EGNN
from kinodata.model.egin import HeteroEGIN

from kinodata import configuration


def make_egnn_model(config: configuration.Config) -> RegressionModel:

    # keyword arguments for the message passing class
    mp_kwargs = {
        "rbf_size": config.rbf_size,
        "interaction_radius": config.interaction_radius,
        "reduce": config.mp_reduce,
    }

    # dirty code
    # wandb does not accept this in the config..

    docking_score_num = 2 if config.add_docking_scores else 0
    edge_attr_size = {
        # 4: single, double, triple, "other"
        ("ligand", "interacts", "ligand"): 4 + docking_score_num,
        ("pocket", "interacts", "ligand"): docking_score_num,
        ("ligand", "interacts", "pocket"): docking_score_num,
    }

    model = RegressionModel(
        config,
        partial(EGNN, message_layer_kwargs=mp_kwargs, edge_attr_size=edge_attr_size),
    )

    return model


def make_egin_model(config: configuration.Config) -> RegressionModel:
    return RegressionModel(config, HeteroEGIN)


def make_data(config, transforms=None) -> Tuple[KinodataDocked, LightningDataset]:
    if transforms is None:
        transforms = []

    if config.perturb_ligand_positions:
        transforms.append(
            PerturbAtomPositions("ligand", std=config.perturb_ligand_positions)
        )
    if config.perturb_pocket_positions:
        transforms.append(
            PerturbAtomPositions("pocket", std=config.perturb_pocket_positions)
        )

    transforms.append(AddDistancesAndInteractions(config.interaction_radius))

    if config.add_docking_scores:
        transforms.extend(
            [
                AddGlobalAttrToEdge(
                    ("pocket", "interacts", "ligand"), ["docking_score", "posit_prob"]
                ),
                AddGlobalAttrToEdge(
                    ("ligand", "interacts", "pocket"), ["docking_score", "posit_prob"]
                ),
                AddGlobalAttrToEdge(
                    ("ligand", "interacts", "ligand"), ["docking_score", "posit_prob"]
                ),
            ]
        )

    transform = None if len(transforms) == 0 else Compose(transforms)

    if config.add_artificial_decoys:
        # cannot implement "ligand substitution" as a standard transform
        dataset = KinodataDockedWithDecoys(transform=transform)
    else:
        dataset = KinodataDocked(transform=transform)

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


def train_regressor(config, fn_model: Callable[..., RegressionModel]):
    logger = WandbLogger(log_model="all")

    _, data_module = make_data(config)
    model = fn_model(config)
    val_checkpoint_callback = ModelCheckpoint(monitor="val_mae", mode="min")
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


if __name__ == "__main__":
    meta_config = configuration.get("meta")
    meta_config = configuration.overwrite_from_file(
        meta_config, "config_regressor_local.yaml"
    )
    config = configuration.get("data", meta_config.model_type, "training")
    config["lr"] = 1e-4
    config["add_docking_scores"] = True
    config["model_type"] = "egnn"
    config = configuration.overwrite_from_file(config, "config_regressor_local.yaml")

    if meta_config.model_type == "egin":
        fn_model = make_egin_model
    if meta_config.model_type == "egnn":
        fn_model = make_egnn_model

    print(config)
    wandb.init(config=config, project="kinodata-docked-rescore")
    train_regressor(config, fn_model)
