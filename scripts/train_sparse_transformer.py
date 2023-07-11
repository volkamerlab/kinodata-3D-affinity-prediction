from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import wandb

import kinodata.configuration as configuration
from kinodata.training import train
from kinodata.model.complex_transformer import ComplexTransformer
from kinodata.types import NodeType, RelationType


def make_model(config: configuration.Config):
    cls = partial(ComplexTransformer, config)
    return config.init(cls)


if __name__ == "__main__":
    configuration.register(
        "sparse_transformer",
        max_num_neighbors=16,
        hidden_channels=256,
        num_attention_blocks=3,
        num_heads=8,
        act="relu",
        edge_attr_size=4,
        ln1=True,
        ln2=True,
        ln3=True,
        graph_norm=False,
        interaction_modes=["covalent"],
    )
    config = configuration.get("data", "training", "sparse_transformer")
    config = config.update_from_file()
    config = config.update_from_args()
    config["need_distances"] = False
    config["perturb_ligand_positions"] = 0.0
    config["perturb_pocket_positions"] = 0.0
    config["perturb_complex_positions"] = 0.1

    config["node_types"] = [NodeType.Complex]

    for key, value in sorted(config.items(), key=lambda i: i[0]):
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["transformer"])
    train(config, fn_model=make_model)
