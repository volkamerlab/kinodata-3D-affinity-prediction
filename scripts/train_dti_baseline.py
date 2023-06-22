from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import wandb

import kinodata.configuration as configuration
from kinodata.model.dti import (
    DTIModel,
    GlobalSumDecoder,
    ResidueTransformer,
    KissimTransformer,
    LigandGINE,
)
from kinodata.data.featurization.residue import known_residues
from kinodata.training import train


class ResidueFeaturization:
    Kissim = "kissim"
    Onehot = "onehot"


def make_dti_model(config):
    if config.residue_featurization == ResidueFeaturization.Onehot:
        ResidueModel = ResidueTransformer
        config["residue_size"] = len(known_residues) + 1
    if config.residue_featurization == ResidueFeaturization.Kissim:
        ResidueModel = KissimTransformer
        config["residue_size"] = 6
    return DTIModel(config, LigandGINE, ResidueModel, GlobalSumDecoder)


if __name__ == "__main__":
    configuration.register(
        "dti_baseline",
        num_layers=3,
        hidden_channels=128,
        act="silu",
        num_attention_blocks=2,
        residue_featurization=ResidueFeaturization.Onehot,
    )

    config = configuration.get("data", "training", "dti_baseline")
    config = config.update_from_args()
    config["need_distances"] = False

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(config=config, project="kinodata-docked-rescore", tags=["dti"])
    train(config, fn_model=make_dti_model)
