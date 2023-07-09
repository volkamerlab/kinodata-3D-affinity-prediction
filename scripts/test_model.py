from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import cuda

import wandb
from kinodata.configuration import from_wandb, register
from kinodata.data.data_module import make_kinodata_module as make_data
from train_ligand_gnn_baseline import make_model as make_baseline_model
from train_egnn import make_egnn_model
from train_dti_baseline import make_dti_model

from wandb_utils import retrieve_best_model_artifact


def main():
    # usage test_model.py --run_path "nextaids/kinodata-docked-rescore/<run_id>"
    test_config = register("test", run_path="nextaids/kinodata-docked-rescore/3nuebccj")
    test_config = test_config.update_from_args()

    # load training run and config
    api = wandb.Api()
    training_run = api.run(test_config.run_path)
    config_used_by_train = from_wandb(training_run)

    if "ligand-only" in training_run.tags:
        make_model = make_baseline_model
    elif "dti" in training_run.tags:
        make_model = make_dti_model
    else:
        make_model = make_egnn_model

    wandb.config.update(config_used_by_train)
    wandb.config.update({"num_workers": 0}, allow_val_change=True)
    wandb.config.update({"source_run": test_config.run_path})

    # convert edge types to tuple st they are hashable
    config_used_by_train["edge_types"] = [
        tuple(et) for et in config_used_by_train["edge_types"]
    ]
    model = make_model(config_used_by_train)

    artifact_dir = retrieve_best_model_artifact(training_run).download()
    ckpt = torch.load(
        Path(artifact_dir) / "model.ckpt", map_location=torch.device("cpu")
    )
    model.load_state_dict(ckpt["state_dict"])
    print(model)
    logger = WandbLogger(log_model="all")

    data_module = make_data(wandb.config)

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        accelerator="gpu" if cuda.is_available() else "cpu",
    )
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    wandb.init(project="kinodata-docked-rescore", tags=["testing"])
    main()
