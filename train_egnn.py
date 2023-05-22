from collections import defaultdict
import sys
from typing import Callable, Dict

from kinodata.model.resolve import resolve_aggregation

# dirty
sys.path.append(".")

from functools import partial

import wandb

from kinodata import configuration
from kinodata.model.egin import HeteroEGIN
from kinodata.model.egnn import EGNN
from kinodata.model.shared.node_embedding import (
    HeteroEmbedding,
    AtomTypeEmbedding,
    FeatureEmbedding,
)
from kinodata.model.complex_mpnn import MessagePassingModel
from kinodata.model.shared.readout import HeteroReadout
from kinodata.training import train
from kinodata.types import EdgeType


def infer_edge_attr_size(config: configuration.Config) -> Dict[EdgeType, int]:
    # dirty code
    # wandb does not accept this in the config..

    docking_score_num = 2 if config.add_docking_scores else 0
    edge_attr_size = defaultdict(int)
    edge_attr_size[("ligand", "interacts", "ligand")] = 4 + docking_score_num
    edge_attr_size[("pocket", "interacts", "ligand")] = docking_score_num
    edge_attr_size[("ligand", "interacts", "pocket")] = docking_score_num

    return edge_attr_size


def make_atom_embedding_cls(
    config: configuration.Config,
) -> Callable[..., MessagePassingModel]:
    emb = {
        "ligand": AtomTypeEmbedding("ligand", hidden_chanels=config.hidden_channels),
        "pocket": AtomTypeEmbedding("pocket", hidden_chanels=config.hidden_channels),
        "pocket_residue": FeatureEmbedding(
            "pocket_residue",
            in_channels=config.num_residue_features,
            hidden_channels=config.hidden_channels,
            act=config.act,
        ),
    }
    return partial(
        HeteroEmbedding,
        **{node_type: emb[node_type] for node_type in config.node_types},
    )


def make_egnn_model(config: configuration.Config) -> MessagePassingModel:

    # keyword arguments for the message passing class
    mp_layer_kwargs = {
        edge_type: {
            "rbf_size": config.rbf_size,
            "interaction_radius": config.residue_interaction_radius
            if "residue" in "".join((edge_type[0], edge_type[1]))
            else config.interaction_radius,
            "reduce": config.mp_reduce,
        }
        for edge_type in config.edge_types
    }

    edge_attr_size = infer_edge_attr_size(config)

    model = MessagePassingModel(
        config,
        embedding_cls=make_atom_embedding_cls(config),
        message_passing_cls=partial(
            EGNN, message_layer_kwargs=mp_layer_kwargs, edge_attr_size=edge_attr_size
        ),
        readout_cls=partial(
            HeteroReadout,
            node_aggregation=resolve_aggregation(config.readout_aggregation_type),
            aggregation_out_channels=config.hidden_channels,
            out_channels=1,
        ),
    )

    return model


if __name__ == "__main__":
    config = configuration.get("data", "egnn", "training")
    config["node_types"] = ["ligand", "pocket_residue"]
    config["edge_types"] = [
        ("ligand", "interacts", "ligand"),
        ("ligand", "interacts", "pocket_residue"),
        ("pocket_residue", "interacts", "ligand"),
    ]

    config["add_docking_scores"] = False
    config["model_type"] = "egnn"
    config = config.update_from_file("config_regressor_local.yaml")
    config = config.update_from_args()

    for key, value in config.items():
        print(f"{key}: {value}")
    wandb.init(config=config, project="kinodata-docked-rescore")
    train(config, fn_model=make_egnn_model)
