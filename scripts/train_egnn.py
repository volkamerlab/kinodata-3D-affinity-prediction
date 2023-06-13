from collections import defaultdict
import sys
from typing import Callable, Dict

from kinodata.model.resolve import resolve_aggregation

# dirty
sys.path.append(".")
sys.path.append("..")

from functools import partial

import torch
from torch.nn import Embedding
import wandb

from kinodata import configuration
from kinodata.model.egnn import EGNN, REGNN
from kinodata.model.shared.node_embedding import (
    HeteroEmbedding,
    AtomTypeEmbedding,
    FeatureEmbedding,
)
from kinodata.model.complex_mpnn import MessagePassingModel
from kinodata.model.shared.readout import HeteroReadout
from kinodata.training import train
from kinodata.types import (
    EdgeType,
    NodeType,
    STRUCTURAL_EDGE_TYPES,
    COVALENT_EDGE_TYPES,
)
from kinodata.data.featurization.atoms import AtomFeatures
from kinodata.data.featurization.bonds import NUM_BOND_TYPES

from wandb_utils import sweep


def infer_edge_attr_size(config: configuration.Config) -> Dict[EdgeType, int]:
    # dirty code
    # wandb does not accept this in the config..

    docking_score_num = 2 if config.add_docking_scores else 0
    edge_attr_size = defaultdict(int)
    for edge_type in COVALENT_EDGE_TYPES:
        edge_attr_size[edge_type] = NUM_BOND_TYPES + docking_score_num
    for edge_type in STRUCTURAL_EDGE_TYPES:
        edge_attr_size[edge_type] = docking_score_num

    return edge_attr_size


def make_atom_embedding_cls(
    config: configuration.Config,
) -> Callable[..., HeteroEmbedding]:
    # TODO
    # somehow set this based on config?
    shared_element_embedding = Embedding(100, config.hidden_channels)
    default_embeddings: Dict[NodeType, torch.nn.Module] = {
        NodeType.Ligand: [
            AtomTypeEmbedding(
                NodeType.Ligand,
                hidden_chanels=config.hidden_channels,
                embedding=shared_element_embedding,
            ),
            FeatureEmbedding(
                NodeType.Ligand,
                in_channels=AtomFeatures.size,
                hidden_channels=config.hidden_channels,
                act=config.act,
            ),
        ],
        NodeType.Pocket: [
            AtomTypeEmbedding(
                NodeType.Pocket,
                hidden_chanels=config.hidden_channels,
                embedding=shared_element_embedding,
            ),
            FeatureEmbedding(
                NodeType.Pocket,
                in_channels=AtomFeatures.size,
                hidden_channels=config.hidden_channels,
                act=config.act,
            ),
        ],
        NodeType.PocketResidue: FeatureEmbedding(
            NodeType.PocketResidue,
            in_channels=config.num_residue_features,
            hidden_channels=config.hidden_channels,
            act=config.act,
        ),
    }
    return partial(
        HeteroEmbedding,
        **{node_type: default_embeddings[node_type] for node_type in config.node_types},
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

    if config.model_type.lower() == "egnn":
        EGNNCls = EGNN
    elif config.model_type.lower() == "rel-egnn":
        EGNNCls = REGNN
    else:
        raise ValueError(config.model_type)

    model = MessagePassingModel(
        config,
        embedding_cls=make_atom_embedding_cls(config),
        message_passing_cls=partial(
            EGNNCls, message_layer_kwargs=mp_layer_kwargs, edge_attr_size=edge_attr_size
        ),
        readout_cls=partial(
            HeteroReadout,
            node_aggregation=resolve_aggregation(config.readout_aggregation_type),
            aggregation_out_channels=config.hidden_channels,
            out_channels=1,
        ),
    )

    return model


def main():
    wandb.init(project="kinodata-docked-rescore")
    config = configuration.get("data", "egnn", "training")
    config["node_types"] = [NodeType.Ligand, NodeType.Pocket]
    config["edge_types"] = COVALENT_EDGE_TYPES + STRUCTURAL_EDGE_TYPES

    config["add_docking_scores"] = False
    config = config.update_from_file("config_regressor_local.yaml")
    config = config.update_from_args()

    config.update(wandb.config)
    wandb.config.update(config)

    for key, value in config.items():
        print(f"{key}: {value}")

    train(config, fn_model=make_egnn_model)


if __name__ == "__main__":
    main()
