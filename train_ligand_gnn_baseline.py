import sys

# dirty
sys.path.append(".")

import wandb

from torch_geometric.nn.models import GIN
from torch_geometric.nn.resolver import aggregation_resolver

from kinodata.model.ligand_gin import LigandGNNBaseline
from kinodata import configuration
from kinodata.training import train


def make_model(config) -> LigandGNNBaseline:
    if config.gnn_type == "gin":
        encoder = GIN(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=config.hidden_channels,
            act=config.act,
            norm="GraphNorm",
        )
    else:
        raise ValueError(config.gnn_type)

    readout = aggregation_resolver(config.readout_type)

    model = LigandGNNBaseline(config, encoder, readout)
    return model


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
    config = config.update_from_file("config_ligand_baseline_local.yaml")
    config = config.update_from_args()

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["ligand-only"])
    train(config, fn_model=make_model)
