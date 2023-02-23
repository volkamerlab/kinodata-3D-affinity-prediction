from pathlib import Path

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from train_ligand_gnn_baseline import make_data, make_model
from kinodata.model.ligand_gin_baseline import LigandGNNBaseline


if __name__ == "__main__":
    run = wandb.init()
    artifact = run.use_artifact(
        "nextaids/kinodata-docked-rescore/model-34nafay5:v37", type="model"
    )
    other_run = artifact.logged_by()
    wandb.config.update(other_run.config)
    wandb.config.update({"num_workers": 0}, allow_val_change=True)
    model = make_model(wandb.config)

    artifact_dir = artifact.download()
    ckpt = torch.load(
        Path(artifact_dir) / "model.ckpt", map_location=torch.device("cpu")
    )
    model.load_state_dict(ckpt["state_dict"])

    logger = WandbLogger(log_model="all")

    _, data_module = make_data(wandb.config)

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        accelerator="cpu",
    )

    trainer.test(model, datamodule=data_module)

    pass
