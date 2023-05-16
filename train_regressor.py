import sys
from typing import Dict

# dirty
sys.path.append(".")

from functools import partial

import wandb

from kinodata import configuration
from kinodata.model.egin import HeteroEGIN
from kinodata.model.egnn import EGNN
from kinodata.model.regression_model import RegressionModel
from kinodata.training import train
from kinodata.types import EdgeType


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


if __name__ == "__main__":
    meta_config = configuration.get("meta")
    meta_config = meta_config.update_from_file("config_regressor_local.yaml")
    config = configuration.get("data", meta_config.model_type, "training")
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
    train(config, fn_model=fn_model)
