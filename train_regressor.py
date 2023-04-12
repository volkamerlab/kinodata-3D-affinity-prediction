from pathlib import Path
import sys
from typing import Callable, Optional, Tuple, Dict

# dirty
sys.path.append(".")

from functools import partial

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
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


def compose(transforms: Optional[list]) -> Optional[Callable]:
    return None if transforms is None else Compose(transforms)


def infer_edge_attr_size(config: configuration.Config) -> Dict[EdgeType, int]:
    # dirty code
    # wandb does not accept this in the config..

    docking_score_num = 2 if config.add_docking_scores else 0
    edge_attr_size = {
        # 4: single, double, triple, "other" bond
        ("ligand", "interacts", "ligand"): 4 + docking_score_num,
        ("pocket", "interacts", "ligand"): docking_score_num,
        ("ligand", "interacts", "pocket"): docking_score_num,
    }

    return edge_attr_size


def make_egnn_model(config: configuration.Config) -> RegressionModel:

    # keyword arguments for the message passing class
    mp_kwargs = {
        "rbf_size": config.rbf_size,
        "interaction_radius": config.interaction_radius,
        "reduce": config.mp_reduce,
    }

    edge_attr_size = infer_edge_attr_size(config)

    model = RegressionModel(
        config,
        partial(EGNN, message_layer_kwargs=mp_kwargs, edge_attr_size=edge_attr_size),
    )

    return model


def make_egin_model(config: configuration.Config) -> RegressionModel:
    edge_attr_size = infer_edge_attr_size(config)
    return RegressionModel(config, partial(HeteroEGIN, edge_attr_size=edge_attr_size))


def make_data(config: configuration.Config, transforms=None) -> LightningDataset:
    if transforms is None:
        transforms = []

    augmentations = []

    if config.perturb_ligand_positions:
        augmentations.append(
            PerturbAtomPositions("ligand", std=config.perturb_ligand_positions)
        )
    if config.perturb_pocket_positions:
        augmentations.append(
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

    train_transform = compose(augmentations + transforms)
    val_transform = compose(transforms)

    if config.add_artificial_decoys:
        dataset_cls = KinodataDockedWithDecoys
    else:
        dataset_cls = KinodataDocked

    if not Path(config.data_split).exists():
        raise FileNotFoundError(config.data_split)

    print(f"Loading split from {config.data_split}..")
    # split that assigns *idents* to train/val/test
    split = Split.from_data_frame(pd.read_csv(config.data_split))
    print("Remapping idents to dataset index..")
    index_mapping = dataset_cls().ident_index_map()
    index_mapping = {int(k): v for k, v in index_mapping.items()}
    split = split.remap_index(index_mapping)

    print("Creating data module:")
    print(f"    split:{split}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    data_module = make_data_module(
        split,
        config.batch_size,
        config.num_workers,
        dataset_cls=dataset_cls,
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
    )

    return data_module


def train_regressor(
    config: configuration.Config, fn_model: Callable[..., RegressionModel]
):
    logger = WandbLogger(log_model="all")

    data_module = make_data(config)
    model = fn_model(config)
    val_checkpoint_callback = ModelCheckpoint(monitor="val/mae", mode="min")
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
    meta_config = configuration.get("meta")
    meta_config = meta_config.update_from_file("config_regressor_local.yaml")
    config = configuration.get("data", meta_config.model_type, "training")
    config["lr"] = 1e-4
    config["add_docking_scores"] = False
    config["model_type"] = "egnn"
    config = config.update_from_file("config_regressor_local.yaml")
    config = config.update_from_args()

    if meta_config.model_type == "egin":
        fn_model = make_egin_model
    if meta_config.model_type == "egnn":
        fn_model = make_egnn_model

    for key, value in config.items():
        print(f"{key}: {value}")
    wandb.init(config=config, project="kinodata-docked-rescore")
    train_regressor(config, fn_model)
