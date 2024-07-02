from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")
import kinodata.configuration as configuration
from kinodata.model.dti import ResidueFeaturization, make_model
from kinodata.training import train
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.add_kissim import AddKissimFingerprint
from kinodata.data.data_module import make_kinodata_module

import wandb

if __name__ == "__main__":
    configuration.register(
        "dti_baseline",
        num_layers=3,
        hidden_channels=128,
        act="silu",
        num_attention_blocks=2,
        residue_featurization=ResidueFeaturization.Kissim,
    )

    config = configuration.get("data", "training", "dti_baseline")
    config = config.update_from_args()
    config = config.update_from_file()
    config["need_distances"] = False

    for key, value in config.items():
        print(f"{key}: {value}")

    wandb.init(
        config=config, project="kinodata-docked-rescore", tags=["dti", "less-features"]
    )
    ott = None
    if config["residue_featurization"] == ResidueFeaturization.Kissim:
        ott = partial(apply_transform_instance_permament, transform=AddKissimFingerprint())
    train(config, fn_model=make_model, fn_data=partial(make_kinodata_module, one_time_transform=ott))