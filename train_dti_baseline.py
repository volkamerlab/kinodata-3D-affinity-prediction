import sys

# dirty
sys.path.append(".")
import wandb

import kinodata.configuration as configuration
from kinodata.model.dti import (
    DTIModel,
    GlobalSumDecoder,
    KissimPocketTransformer,
    LigandGINE,
)
from kinodata.training import train


def make_dti_model(config):
    return DTIModel(config, LigandGINE, KissimPocketTransformer, GlobalSumDecoder)


if __name__ == "__main__":
    configuration.register(
        "dti_baseline",
        num_layers=3,
        hidden_channels=128,
        act="silu",
        num_attention_blocks=2,
        kissim_size=12,
    )

    config = configuration.get("data", "training", "dti_baseline")
    config = config.update_from_args()
    config["need_distances"] = False

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["dti"])
    train(config, fn_model=make_dti_model)
