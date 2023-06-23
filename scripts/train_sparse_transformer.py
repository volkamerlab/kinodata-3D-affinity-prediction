from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import wandb

import kinodata.configuration as configuration
from kinodata.training import train
from kinodata.model.complex_transformer import ComplexTransformer


def make_model(config: configuration.Config):
    cls = partial(ComplexTransformer, config)
    return config.init(cls)


if __name__ == "__main__":
    configuration.register(
        "sparse_transformer",
        hidden_channels=128,
        num_attention_blocks=3,
        num_heads=8,
        act="relu",
        edge_attr_size=4,
        ln1=True,
        ln2=False,
        ln3=True,
        graph_norm=True,
    )
    config = configuration.get("data", "training", "sparse_transformer")
    config = config.update_from_file("config_regressor_local.yaml")
    config["need_distances"] = False

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["transformer"])
    train(config, fn_model=make_model)
